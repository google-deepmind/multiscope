// Package events implements functions to process events.
package events

import (
	"log"
	"sync"

	"multiscope/internal/server/core"
	pb "multiscope/protos/tree_go_proto"

	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/types/known/anypb"
)

// Registry maintains a registry of node paths and their Queues.
type Registry struct {
	mtx sync.RWMutex

	nodeEvents map[core.Key][]*Queue
	allEvents  []*Queue
}

// NewRegistry returns a new registry to process events.
func NewRegistry() *Registry {
	return &Registry{
		nodeEvents: make(map[core.Key][]*Queue),
	}
}

func (reg *Registry) fetchQueues(event *pb.Event) (qs []*Queue) {
	reg.mtx.RLock()
	defer reg.mtx.RUnlock()
	defer func() {
		qs = append([]*Queue{}, qs...)
	}()

	if event.Path != nil {
		nodeQueues := reg.nodeEvents[core.ToKey(event.Path.Path)]
		if len(nodeQueues) > 0 {
			return nodeQueues
		}
	}
	return reg.allEvents
}

// Process the incoming frontend event.
func (reg *Registry) Process(event *pb.Event) {
	if event.Payload == nil {
		event.Payload = &anypb.Any{}
	}
	for _, s := range reg.fetchQueues(event) {
		if !s.write(event) {
			log.Printf("Dropped event with type %q for path %s because buffer full", event.GetPayload().GetTypeUrl(), prototext.Format(event.GetPath()))
		}
	}
}

func removeFrom(q *Queue, qs []*Queue) []*Queue {
	for i, qi := range qs {
		if qi != q {
			continue
		}
		qs = append(qs[:i], qs[i+1:]...)
	}
	return qs
}

func (reg *Registry) unsubscribe(queue *Queue) {
	reg.mtx.Lock()
	defer reg.mtx.Unlock()

	if queue.nodeQueue {
		reg.nodeEvents[queue.key] = removeFrom(queue, reg.nodeEvents[queue.key])
	} else {
		reg.allEvents = removeFrom(queue, reg.allEvents)
	}
}

// NewQueue creates a new queue to process all events not captured by a node in the tree.
// Delete needs to be call on the queue to avoid leaks.
func (reg *Registry) NewQueue(cb Callback) *Queue {
	reg.mtx.Lock()
	defer reg.mtx.Unlock()
	queue := newQueue(reg, cb, false, "")
	reg.allEvents = append(reg.allEvents, queue)
	return queue
}

// NewQueueForPath returns a queue to process events for a given path in the tree.
func (reg *Registry) NewQueueForPath(path []string, cb Callback) *Queue {
	reg.mtx.Lock()
	defer reg.mtx.Unlock()
	key := core.ToKey(path)
	queue := newQueue(reg, cb, true, key)
	reg.nodeEvents[key] = append(reg.nodeEvents[key], queue)
	return queue
}
