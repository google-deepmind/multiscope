// Copyright 2023 Google LLC
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

package events

import (
	"log"
	"multiscope/internal/server/core"
	pb "multiscope/protos/tree_go_proto"
	"sync"
)

// Queue implements a bounded queue that only retains the last `queueSize`
// events.
type Queue struct {
	wg        sync.WaitGroup
	reg       *Registry
	cb        Callback
	key       core.Key
	nodeQueue bool

	toProcess chan *pb.Event
}

// newQueue returns a new Queue.
func newQueue(reg *Registry, cb Callback, nodeQueue bool, key core.Key) *Queue {
	q := &Queue{
		reg:       reg,
		cb:        cb,
		key:       key,
		nodeQueue: nodeQueue,
		toProcess: make(chan *pb.Event, 100),
	}
	q.wg.Add(1)
	go q.process()
	return q
}

func (q *Queue) process() {
	for ev := range q.toProcess {
		err := q.cb(ev)
		if err != nil {
			// TODO(degris): surface callback errors in the UI
			log.Printf("event callback for path %q returned error: %v", q.key, err)
		}
	}
	q.wg.Done()
}

// write writes an event into r.next, or r.queue if r.next is full. write can be
// called concurrently with Next and other calls to write. Returns true if the
// write caused an event to be dropped.
func (q *Queue) write(ev *pb.Event) bool {
	select {
	case q.toProcess <- ev:
		return true
	default:
		return false
	}
}

// Delete unregisters the queue and terminate the Go routine.
func (q *Queue) Delete() {
	q.reg.unsubscribe(q)
	close(q.toProcess)
	q.wg.Wait()
}
