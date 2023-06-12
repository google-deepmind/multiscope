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

package scope

import (
	"multiscope/internal/server/core"
	"multiscope/internal/server/events"
	"multiscope/internal/server/pathlog"
	"multiscope/internal/server/root"
	"multiscope/internal/server/treeservice"
	"multiscope/internal/server/writers"
	"net/url"
	"sync"
	"time"
)

const activePathThreshold = 1 * time.Minute

type singleton struct {
	treeID core.TreeID
	state  *treeservice.State
	mut    sync.Mutex

	dispatcher treeservice.EventDispatcher
}

var (
	_ treeservice.URLToState   = (*singleton)(nil)
	_ treeservice.StateFactory = (*singleton)(nil)
)

func (s *singleton) newState() {
	s.state = treeservice.NewState(
		s,
		root.NewRoot(),
		events.NewRegistry(),
		pathlog.NewPathLog(activePathThreshold),
		s.dispatcher)
}

func (s *singleton) TreeID() core.TreeID {
	return s.treeID
}

func (s *singleton) NewState() *treeservice.State {
	s.mut.Lock()
	defer s.mut.Unlock()
	s.newState()
	return s.state
}

func (s *singleton) ToState(url *url.URL) (*treeservice.State, error) {
	s.mut.Lock()
	defer s.mut.Unlock()
	if s.state == nil {
		s.newState()
	}
	return s.state, nil
}

func (s *singleton) Delete(core.TreeID) {}

// NewSingleton starts a new Multiscope server where the state is shared across all clients.
func NewSingleton() *treeservice.TreeServer {
	state := &singleton{treeID: 1}
	srv := treeservice.New(writers.All(), state)
	state.dispatcher = srv
	return srv
}
