// Copyright 2023 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package events_test

import (
	"sync"
	"testing"

	"multiscope/internal/server/events"
	pb "multiscope/protos/tree_go_proto"
)

type TestCase struct {
	pathsToRegister []string
	capture         bool
	pathsToProcess  []string
	wantCount       int

	mut sync.Mutex
	got []*pb.Event
}

func (tc *TestCase) process(ev *pb.Event) error {
	tc.mut.Lock()
	defer tc.mut.Unlock()
	tc.got = append(tc.got, ev)
	return nil
}

func TestEventsDispatchedToSubscribers(t *testing.T) {
	testCases := []*TestCase{
		{
			pathsToRegister: []string{"node"},
			pathsToProcess:  []string{"node"},
			wantCount:       1,
		},
		{
			pathsToRegister: []string{"node"},
			pathsToProcess:  []string{"completely_different_node"},
			wantCount:       0,
		},
		{
			pathsToRegister: []string{"node", "node"},
			pathsToProcess:  []string{"node"},
			wantCount:       2,
		},
		{
			pathsToRegister: []string{""},
			pathsToProcess:  []string{"node"},
			wantCount:       1,
		},
		{
			pathsToRegister: []string{"node"},
			pathsToProcess:  []string{"node", "node"},
			wantCount:       2,
		},
		{
			pathsToRegister: []string{"node"},
			capture:         true,
			pathsToProcess:  []string{"node", "node"},
			wantCount:       2,
		},
	}

	for i, testCase := range testCases {
		reg := events.NewRegistry()

		var queues []*events.Queue
		for _, path := range testCase.pathsToRegister {
			var q *events.Queue
			if len(path) == 0 {
				q = reg.NewQueue(testCase.process)
			} else {
				q = reg.NewQueueForPath([]string{path}, testCase.process)
			}
			queues = append(queues, q)
		}

		for _, path := range testCase.pathsToProcess {
			reg.Process(&pb.Event{
				Path: &pb.NodePath{Path: []string{path}},
			})
		}

		for _, q := range queues {
			q.Delete()
		}

		gotCount := len(testCase.got)
		if gotCount != testCase.wantCount {
			t.Errorf("testcase %d: wrong number of events published: got %d want %d", i, gotCount, testCase.wantCount)
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
			unsubscribeIdx: []int{0},
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

	reg := events.NewRegistry()
	for _, testCase := range testCases {
		var queues []*events.Queue
		got := make([][]*pb.Event, len(testCase.subscribe))
		for i, s := range testCase.subscribe {
			i := i
			cb := func(ev *pb.Event) error {
				got[i] = append(got[i], ev)
				return nil
			}
			q := reg.NewQueueForPath([]string{s}, cb)
			queues = append(queues, q)
		}
		for _, idx := range testCase.unsubscribeIdx {
			queues[idx].Delete()
			queues[idx] = nil
		}
		for _, p := range testCase.publish {
			reg.Process(&pb.Event{
				Path: &pb.NodePath{Path: []string{p}},
			})
		}
		for _, q := range queues {
			if q != nil {
				q.Delete()
			}
		}
		for i := range testCase.subscribe {
			if len(got[i]) != testCase.expect[i] {
				t.Errorf("wrong number of events published for path %q, got: %d wanted %d at subscriber idx %d", testCase.subscribe[i], len(got[i]), testCase.expect[i], i)
			}
		}
	}
}
