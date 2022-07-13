package events_test

import (
	"fmt"
	"testing"

	"multiscope/internal/server/events"
	pb "multiscope/protos/tree_go_proto"
)

type TestCase struct {
	pathsToRegister []string
	capture         bool
	pathsToProcess  []string
	wantCount       int
	eventQueue      []*events.EventQueue
}

func consume(queue *events.EventQueue) (events []*pb.Event, err error) {
	for {
		var e *pb.Event
		e, err = queue.Poll()
		if err != nil {
			return nil, err
		}
		if e == nil {
			return
		}
		events = append(events, e)
	}
}

func TestEventsDispatchedToSubscribers(t *testing.T) {
	testCases := []TestCase{
		TestCase{
			pathsToRegister: []string{"node"},
			pathsToProcess:  []string{"node"},
			wantCount:       1,
		},
		TestCase{
			pathsToRegister: []string{"node"},
			pathsToProcess:  []string{"completely_different_node"},
			wantCount:       0,
		},
		TestCase{
			pathsToRegister: []string{"node", "node"},
			pathsToProcess:  []string{"node"},
			wantCount:       2,
		},
		TestCase{
			pathsToRegister: []string{""},
			pathsToProcess:  []string{"node"},
			wantCount:       1,
		},
		TestCase{
			pathsToRegister: []string{"node"},
			pathsToProcess:  []string{"node", "node"},
			wantCount:       2,
		},
		TestCase{
			pathsToRegister: []string{"node"},
			capture:         true,
			pathsToProcess:  []string{"node", "node"},
			wantCount:       2,
		},
	}

	for i, testCase := range testCases {
		registry := events.NewRegistry()

		for _, path := range testCase.pathsToRegister {
			testCases[i].eventQueue = append(testCases[i].eventQueue, registry.Subscribe([]string{path}, ""))
		}

		for _, path := range testCase.pathsToProcess {
			registry.Process(&pb.Event{
				Path: &pb.NodePath{Path: []string{path}},
			})
		}

		gotCount := 0
		for _, queue := range testCases[i].eventQueue {
			events, err := consume(queue)
			if err != nil {
				t.Error(err)
			}
			gotCount += len(events)
		}

		if gotCount != testCase.wantCount {
			t.Errorf("wrong number of events published: got %d want %d", gotCount, testCase.wantCount)
		}
	}
}

func TestUnsubscribedQueueStopsReceivingEvents(t *testing.T) {
	testCases := []struct {
		subscribe      []string
		unsubscribeIdx []int
		publish        []string
		expect         []int
	}{
		{
			subscribe:      []string{"A", "B"},
			unsubscribeIdx: []int{0},
			publish:        []string{"A", "B"},
			expect:         []int{0, 1},
		},
		{
			subscribe:      []string{"A", "B"},
			unsubscribeIdx: []int{0, 0},
			publish:        []string{"A", "B"},
			expect:         []int{0, 1},
		},
		{
			subscribe:      []string{"A", "A", "B"},
			unsubscribeIdx: []int{0, 2},
			publish:        []string{"A", "B"},
			expect:         []int{0, 1, 0},
		},
	}

	registry := events.NewRegistry()
	for _, testCase := range testCases {
		var subscribers []*events.EventQueue
		for _, s := range testCase.subscribe {
			subscribers = append(subscribers, registry.Subscribe([]string{s}, ""))
		}
		for _, idx := range testCase.unsubscribeIdx {
			registry.Unsubscribe(subscribers[idx])
		}
		for _, p := range testCase.publish {
			registry.Process(&pb.Event{
				Path: &pb.NodePath{Path: []string{p}},
			})
		}
		for i, s := range subscribers {
			events, err := consume(s)
			if err != nil {
				t.Error(err)
			}
			if len(events) != testCase.expect[i] {
				t.Errorf("wrong number of events published for path %q, got: %d wanted %d at subscriber idx %d", testCase.subscribe[i], len(events), testCase.expect[i], i)
			}
		}
	}
}

func TestPublishRetainsLastQueueSizeEvents(t *testing.T) {
	registry := events.NewRegistry()
	var allEvents []*pb.Event
	queue := registry.Subscribe([]string{}, "")

	for i := 0; i < events.EventQueueSize*2; i++ {
		allEvents = append(allEvents, &pb.Event{
			Path: &pb.NodePath{Path: []string{fmt.Sprintf("%d", i)}},
		})
		registry.Process(allEvents[i])
	}

	publishedEvents, err := consume(queue)
	if err != nil {
		t.Error(err)
	}
	if len(publishedEvents) != events.EventQueueSize {
		t.Errorf("len(publishedEvents) = %d, wanted %d", len(publishedEvents), events.EventQueueSize)
	}
	for i := 0; i < events.EventQueueSize; i++ {
		expected := allEvents[len(allEvents)-events.EventQueueSize+i]
		if publishedEvents[i] != expected {
			t.Errorf("publishedEvents[%d] = %v, expected %v", i, publishedEvents[i], expected)
		}
	}
}

func TestUnsubscribeReturnsNilFromNext(t *testing.T) {
	registry := events.NewRegistry()
	ch := make(chan *pb.Event)
	queue := registry.Subscribe([]string{}, "")
	go func() {
		for {
			e, err := queue.Next()
			if err != nil {
				t.Error(err)
			}
			ch <- e
		}
	}()
	registry.Process(&pb.Event{
		Path: &pb.NodePath{Path: []string{}},
	})
	if e1 := <-ch; e1 == nil {
		t.Errorf("queue.Next() = nil, expected non-nil event before Unsubscribe")
	}
	registry.Unsubscribe(queue)
	if e2 := <-ch; e2 != nil {
		t.Errorf("queue.Next() is non-nil (%v), expected nil after Unsubscribe", e2)
	}
}
