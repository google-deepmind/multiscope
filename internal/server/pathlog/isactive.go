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

package pathlog

import (
	"sync"

	"multiscope/internal/server/core"
	pb "multiscope/protos/tree_go_proto"
)

type activePathSubscriber struct {
	mut     sync.Mutex
	actives map[core.Key]bool
}

func newActivePathSubscriber(sub chan *pb.ActivePathsReply) *activePathSubscriber {
	a := &activePathSubscriber{
		actives: make(map[core.Key]bool),
	}
	go func() {
		for rep := range sub {
			a.update(rep)
		}
	}()
	return a
}

// IsActive returns if a path is active or not.
func (a *activePathSubscriber) IsActive(key core.Key) bool {
	a.mut.Lock()
	defer a.mut.Unlock()
	return a.actives[key]
}

func (a *activePathSubscriber) update(paths *pb.ActivePathsReply) {
	a.mut.Lock()
	defer a.mut.Unlock()
	for k := range a.actives {
		delete(a.actives, k)
	}
	for _, path := range paths.GetPaths() {
		a.actives[core.ToKey(path.GetPath())] = true
	}
}
