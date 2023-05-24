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

package remote

import (
	"context"
	"fmt"
	"sync"
	"time"

	"multiscope/internal/server/core"
	pb "multiscope/protos/tree_go_proto"
	pbgrpc "multiscope/protos/tree_go_proto"
)

// ActiveCallback is called when the state of the path for which it is registered changes.
type ActiveCallback func(key Path, active bool)

// Active maintains a list of active paths.
type Active struct {
	mut sync.Mutex
	clt pbgrpc.Tree_ActivePathsClient

	lastActives map[core.Key]bool
	callbacks   map[core.Key][]ActiveCallback
}

func newActive(clt pbgrpc.TreeClient) (*Active, error) {
	ctx := context.Background()
	activeClient, err := clt.ActivePaths(ctx, &pb.ActivePathsRequest{})
	if err != nil {
		return nil, err
	}
	a := &Active{
		clt:         activeClient,
		lastActives: make(map[core.Key]bool),
		callbacks:   make(map[core.Key][]ActiveCallback),
	}
	go a.listenToServer()
	return a, nil
}

func (a *Active) listenToServer() {
	for {
		rep, err := a.clt.Recv()
		if err != nil {
			fmt.Printf("cannot receive active path message: %v\n", err)
			time.Sleep(time.Minute)
		}
		actives := make(map[core.Key]bool)
		for _, path := range rep.GetPaths() {
			actives[core.ToKey(path.GetPath())] = true
		}
		a.updateCallbacks(actives)
	}
}

func (a *Active) updateCallbacks(actives map[core.Key]bool) {
	a.mut.Lock()
	defer a.mut.Unlock()

	for key := range a.lastActives {
		if actives[key] {
			// The path was already active: remove the key from actives
			// (to avoid calling the callback for nothing).
			delete(actives, key)
			continue
		}
		// The path is not active anymore: remove the key.
		delete(a.lastActives, key)
		a.call(key, false)
	}
	// Key remaining in actives are all the paths that are active now and were not before.
	for key := range actives {
		a.lastActives[key] = true
		a.call(key, true)
	}
}

func (a *Active) call(key core.Key, active bool) {
	for _, cb := range a.callbacks[key] {
		cb(key.Split(), active)
	}
}

// Register a callback to call when the state of a path changes.
// Note that registering a callback will trigger a call to that callback.
func (a *Active) Register(path Path, cb ActiveCallback) {
	a.mut.Lock()
	defer a.mut.Unlock()

	key := core.ToKey(path)
	a.callbacks[key] = append(a.callbacks[key], cb)
	cb(path, a.lastActives[key])
}
