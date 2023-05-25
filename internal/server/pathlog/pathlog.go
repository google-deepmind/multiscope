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

// Package pathlog keeps track of when paths have been requested.
package pathlog

import (
	"log"
	"sort"
	"sync"
	"time"

	"multiscope/internal/server/core"
	pb "multiscope/protos/tree_go_proto"
)

// PathLog keeps a map from paths to the last time there were requested.
type PathLog struct {
	*activePathSubscriber
	activeThreshold   time.Duration
	toUpdate          chan *pb.NodeDataRequest
	toDispatchCurrent chan struct{}
	pathToTime        map[core.Key]time.Time
	lastActive        map[core.Key]bool

	listenersMutex sync.Mutex
	listeners      map[chan *pb.ActivePathsReply]bool

	forwardMutex sync.Mutex
	forwards     map[core.Key][]core.Key
}

// NewPathLog returns a new instance of PathLog.
func NewPathLog(activeThreshold time.Duration) *PathLog {
	pl := &PathLog{
		activeThreshold:   activeThreshold,
		toUpdate:          make(chan *pb.NodeDataRequest, 10),
		toDispatchCurrent: make(chan struct{}, 10),
		pathToTime:        make(map[core.Key]time.Time),
		listeners:         make(map[chan *pb.ActivePathsReply]bool),
		forwards:          make(map[core.Key][]core.Key),
	}
	go pl.update()
	go pl.forceUpdate()
	pl.activePathSubscriber = newActivePathSubscriber(pl.Subscribe())
	return pl
}

func (pl *PathLog) forceUpdate() {
	for {
		time.Sleep(pl.activeThreshold / 2)
		pl.toUpdate <- &pb.NodeDataRequest{}
	}
}

func equal(a, b map[core.Key]bool) bool {
	if len(a) != len(b) {
		return false
	}
	for ai := range a {
		if !b[ai] {
			return false
		}
	}
	return true
}

func (pl *PathLog) update() {
	for {
		var req *pb.NodeDataRequest
		select {
		case <-pl.toDispatchCurrent:
			pl.dispatch()
			continue
		case req = <-pl.toUpdate:
		}
		pl.recordLastRequest(req)
		currentActive := pl.computeActive()
		if equal(pl.lastActive, currentActive) {
			continue
		}
		pl.lastActive = currentActive
		pl.dispatch()
	}
}

// Forward forwards the activity of a path to another.
func (pl *PathLog) Forward(to, from *core.Path) {
	pl.forwardMutex.Lock()
	defer pl.forwardMutex.Unlock()

	fromKey := from.ToKey()
	pl.forwards[fromKey] = append(pl.forwards[fromKey], to.ToKey())
}

// Subscribe returns a channel to listen to the changes of the list of active paths.
func (pl *PathLog) Subscribe() chan *pb.ActivePathsReply {
	pl.listenersMutex.Lock()
	defer pl.listenersMutex.Unlock()

	c := make(chan *pb.ActivePathsReply, 100)
	pl.listeners[c] = true
	return c
}

// Unsubscribe removes the channel from listening to the changes of the list of active paths.
func (pl *PathLog) Unsubscribe(c chan *pb.ActivePathsReply) {
	pl.listenersMutex.Lock()
	defer pl.listenersMutex.Unlock()

	delete(pl.listeners, c)
	close(c)
}

func (pl *PathLog) getListeners() []chan *pb.ActivePathsReply {
	pl.listenersMutex.Lock()
	defer pl.listenersMutex.Unlock()
	ls := make([]chan *pb.ActivePathsReply, 0, len(pl.listeners))
	for l := range pl.listeners {
		ls = append(ls, l)
	}
	return ls
}

func toSorted(actives map[core.Key]bool) []core.Key {
	keys := make([]core.Key, 0, len(actives))
	for k := range actives {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		return keys[i] < keys[j]
	})
	return keys
}

func (pl *PathLog) dispatch() {
	rep := &pb.ActivePathsReply{}
	// Sort pl.lastActive for debugging and testing.
	for _, key := range toSorted(pl.lastActive) {
		rep.Paths = append(rep.GetPaths(), &pb.NodePath{
			Path: key.Split(),
		})
	}
	for _, c := range pl.getListeners() {
		select {
		case c <- rep:
		default:
			log.Printf("Dropped active paths dispatch because buffer full")
		}
	}
}

func (pl *PathLog) recordLastRequest(req *pb.NodeDataRequest) {
	reqs := req.GetReqs()
	now := time.Now()
	for _, req := range reqs {
		if req == nil {
			continue
		}
		path := req.Path
		if path == nil {
			continue
		}
		pl.pathToTime[core.ToKey(path.GetPath())] = now
	}
}

func (pl *PathLog) addKeyAndForwards(keys map[core.Key]bool, current core.Key) {
	if keys[current] {
		return
	}
	keys[current] = true
	for _, forward := range pl.forwards[current] {
		pl.addKeyAndForwards(keys, forward)
	}
}

func (pl *PathLog) computeActive() map[core.Key]bool {
	pl.forwardMutex.Lock()
	defer pl.forwardMutex.Unlock()

	keys := make(map[core.Key]bool)
	now := time.Now()
	for key, last := range pl.pathToTime {
		if now.Sub(last) > pl.activeThreshold {
			continue
		}
		pl.addKeyAndForwards(keys, key)
	}
	return keys
}

// Dispatch records the list of paths from the request and dispatch a list of active paths if such list has change.
// If the request is nil, a dispatch is forced.
func (pl *PathLog) Dispatch(req *pb.NodeDataRequest) {
	pl.toUpdate <- req
}

// DispatchCurrent forces a dispatch to all listeners of the current list of active paths.
func (pl *PathLog) DispatchCurrent() {
	pl.toDispatchCurrent <- struct{}{}
}
