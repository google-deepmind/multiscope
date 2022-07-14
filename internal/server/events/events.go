// Package events implements functions to process events.
package events

import (
	"log"
	"strings"
	"sync"

	"multiscope/internal/server/core"
	pb "multiscope/protos/tree_go_proto"

	"google.golang.org/protobuf/types/known/anypb"
)

// EventQueueSize is the maximum number of events that EventQueue will hold.
// This value is chosen as follows:
//  1. For gamepad and mouse events, we want a small queue size (close to 1)
//     since buffering indicates that readers can't keep up with writers.
//  2. For keyboard events, we want a queue size that is able to keep up with
//     human typing speed. At 10 characters per second this will keep up with
//     5 seconds of unprocessed typing.
//
// In the future we may want separate queue sizes for different cases.
var EventQueueSize = 50

type (
	// Callback signature definition for a function that will receive
	// notifications about events invoked in the frontend.
	Callback interface {
		// Process an event. Should return true to stop the propagation of the event.
		Process(*pb.Event) (bool, error)
	}

	// CallbackF implements the Callback interface with a function.
	CallbackF func(*pb.Event) (bool, error)
)

// Process an event by calling the function. Should return true to stop the propagation of the event.
func (cb CallbackF) Process(ev *pb.Event) (bool, error) {
	return cb(ev)
}

// FilterProto returns a callback filtering events only for a given type of protocol buffers.
func (cb CallbackF) FilterProto(url string) Callback {
	curl := coreURL(url)
	return CallbackF(func(ev *pb.Event) (bool, error) {
		if coreURL(ev.GetPayload().GetTypeUrl()) != curl {
			return false, nil
		}
		return cb(ev)
	})
}

// EventQueue implements a bounded queue that only retains the last `queueSize`
// events.
type EventQueue struct {
	mtx           *sync.Mutex
	queue         []*pb.Event
	queueSize     int
	closed        bool
	notEmpty      *sync.Cond
	typeURLFilter string
}

// newQueue returns a new EventQueue.
func newQueue(queueSize int, typeURLFilter string) *EventQueue {
	mtx := &sync.Mutex{}
	return &EventQueue{
		mtx:           mtx,
		queue:         make([]*pb.Event, 0, queueSize),
		queueSize:     queueSize,
		notEmpty:      sync.NewCond(mtx),
		typeURLFilter: typeURLFilter,
	}
}

// Next waits for the next Event e and returns (&e, nil). If no more events are
// possible, it returns (nil, err) with err set to any error that occurred (nil
// if no error).
// Next can be safely called from multiple goroutines, but each event will only
// be delivered to one waiting goroutine.
func (q *EventQueue) Next() (*pb.Event, error) {
	q.mtx.Lock()
	defer q.mtx.Unlock()
	// Loop needed because q.notEmpty.L is not locked when q.notEmpty.Wait()
	// resumes.
	for {
		if q.closed {
			return nil, nil
		}
		if len(q.queue) != 0 {
			break
		}
		// Releases q.mtx immediately and reacquires it at some point before
		// returning.
		q.notEmpty.Wait()
	}
	e := q.queue[0]
	q.queue = q.queue[1:]
	return e, nil
}

// Poll checks for the next Event e and returns (&e, nil) if an event is
// available. If no events are available, it returns (nil, err) with err set to
// any error that occurred (nil if no error).
// Poll can be safely called from multiple goroutines, but each event will only
// be delivered to one waiting goroutine.
func (q *EventQueue) Poll() (*pb.Event, error) {
	q.mtx.Lock()
	defer q.mtx.Unlock()
	if q.closed {
		return nil, nil
	}
	if len(q.queue) == 0 {
		return nil, nil
	}
	e := q.queue[0]
	q.queue = q.queue[1:]
	return e, nil
}

// write writes an event into r.next, or r.queue if r.next is full. write can be
// called concurrently with Next and other calls to write. Returns true if the
// write caused an event to be dropped.
func (q *EventQueue) write(event *pb.Event) bool {
	if q.typeURLFilter != "" && coreURL(event.GetPayload().GetTypeUrl()) != q.typeURLFilter {
		return false
	}
	q.mtx.Lock()
	defer q.mtx.Unlock()
	defer q.notEmpty.Signal()
	q.queue = append(q.queue, event)
	if len(q.queue) == q.queueSize+1 {
		// drop the oldest event
		q.queue = q.queue[1:]
		return true
	}
	return false
}

// close ensures any goroutines waiting on `r.Next` return immediately.
func (q *EventQueue) close() {
	q.mtx.Lock()
	defer q.mtx.Unlock()
	q.closed = true
	q.notEmpty.Broadcast()
}

func coreURL(url string) string {
	if !strings.HasPrefix(url, "type.google") {
		return url
	}
	sp := strings.Split(url, "/")
	if len(sp) != 2 {
		return url
	}
	return sp[1]
}

// Registerer registers a callback given a path.
type Registerer interface {
	Register(path []string, cbF func() Callback) (Callback, error)
}

// Registry maintains a registry of node paths and their EventQueues.
type Registry struct {
	mtx               sync.RWMutex
	pathToSubscribers map[core.Key][]*EventQueue
	pathToCallback    map[core.Key]Callback
	subscriberToPath  map[*EventQueue]core.Key
}

var _ Registerer = (*Registry)(nil)

// NewRegistry instantiates a Registry structure.
func NewRegistry() *Registry {
	return &Registry{
		pathToSubscribers: make(map[core.Key][]*EventQueue),
		pathToCallback:    make(map[core.Key]Callback),
		subscriberToPath:  make(map[*EventQueue]core.Key),
	}
}

func (h *Registry) getSubscribers(key core.Key) []*EventQueue {
	h.mtx.RLock()
	defer h.mtx.RUnlock()
	chs, ok := h.pathToSubscribers[key]
	if !ok {
		return nil
	}
	return append([]*EventQueue{}, chs...)
}

// Process the incoming frontend event.
func (h *Registry) Process(event *pb.Event) {
	if event.Path == nil {
		event.Path = &pb.NodePath{}
	}
	if event.Payload == nil {
		event.Payload = &anypb.Any{}
	}
	pathKey := core.ToKey(event.GetPath().GetPath())
	subscribers := h.getSubscribers(pathKey)
	// Always publish to global subscribers.
	if pathKey != "" {
		subscribers = append(subscribers, h.getSubscribers("")...)
	}
	for _, s := range subscribers {
		if s.write(event) {
			log.Printf("Dropped event with type %q for path %q because buffer full", event.GetPayload().GetTypeUrl(), event.GetPath().GetPath())
		}
	}
}

// Unsubscribe the provided EventQueue from events. This method is idempotent
// and may be called multiple times, only the first call will have any effect.
func (h *Registry) Unsubscribe(queue *EventQueue) {
	h.mtx.Lock()
	defer h.mtx.Unlock()
	path, ok := h.subscriberToPath[queue]
	if !ok {
		return
	}
	delete(h.subscriberToPath, queue)
	subscribers := make([]*EventQueue, 0, len(h.pathToSubscribers[path])-1)
	// Remove the unsubscribed queue from the list of subscribers for this path.
	for _, s := range h.pathToSubscribers[path] {
		if queue == s {
			s.close()
			continue
		}
		subscribers = append(subscribers, s)
	}
	h.pathToSubscribers[path] = subscribers
}

// Subscribe returns an EventQueue subscribed to events for a node at the
// specified path. If typeURLFilter is not empty, only events with a matching
// type url will be written. The queue is buffered, events may be dropped if not
// processed fast enough.
// Callers must call h.Unsubscribe when done with the EventQueue to avoid
// leaking resources.
func (h *Registry) Subscribe(path []string, typeURLFilter string) *EventQueue {
	h.mtx.Lock()
	defer h.mtx.Unlock()
	queue := newQueue(EventQueueSize, typeURLFilter)
	pathKey := core.ToKey(path)
	h.pathToSubscribers[pathKey] = append(h.pathToSubscribers[pathKey], queue)
	h.subscriberToPath[queue] = pathKey
	return queue
}

func (h *Registry) fetchCallback(path []string, cbF func() Callback) (Callback, bool) {
	h.mtx.Lock()
	defer h.mtx.Unlock()
	pathKey := core.ToKey(path)
	cb := h.pathToCallback[pathKey]
	if cb != nil {
		return cb, false
	}
	cb = cbF()
	h.pathToCallback[pathKey] = cb
	return cb, true
}

// Register an event processing callback for a data node identified by
// the specified path. The callback will run in its own goroutine.
//
// If no callback has been registered to the path, then a new callback is created
// using the factory function provided. If a callback has already been registered
// to the path, then it is returned.
//
// This method does not surface errors from the Callback. Prefer using Subscribe
// instead.
func (h *Registry) Register(path []string, cbF func() Callback) (Callback, error) {
	cb, isNew := h.fetchCallback(path, cbF)
	if !isNew {
		return cb, nil
	}
	queue := h.Subscribe(path, "")
	go func() {
		defer h.Unsubscribe(queue)
		for {
			event, err := queue.Next()
			if err != nil {
				log.Printf("fetching event for path '%v' returned error: %v", path, err)
			}
			if event == nil {
				return
			}
			// TODO(vikrantvarma): `captured` return arg is ignored, remove.
			if _, err := cb.Process(event); err != nil {
				// TODO(vikrantvarma): surface callback errors in the UI
				log.Printf("event callback for path %q returned error: %v", path, err)
			}
		}
	}()
	return cb, nil
}
