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

package treeservice

import (
	"multiscope/internal/server/core"
	"multiscope/internal/server/events"
	"multiscope/internal/server/pathlog"

	"google.golang.org/protobuf/proto"
)

type (
	// StateFactory creates states associated to a given ID.
	StateFactory interface {
		// TreeID returns the ID to which the factory is associated to.
		TreeID() core.TreeID
		// State returns a new state.
		NewState() *State
	}

	// EventDispatcher dispatches events to the experiment.
	EventDispatcher interface {
		// Dispatch sends an event to the experiment.
		Dispatch(path *core.Path, ev proto.Message) error
	}

	// State is the state of a server.
	State struct {
		factory StateFactory

		root        core.Root
		events      *events.Registry
		activePaths *pathlog.PathLog
		dispatcher  EventDispatcher
	}
)

// NewState returns a server state shared across all clients.
func NewState(factory StateFactory, root core.Root, evnts *events.Registry, activePaths *pathlog.PathLog, dispatcher EventDispatcher) *State {
	return &State{
		factory:     factory,
		root:        root,
		events:      evnts,
		activePaths: activePaths,
		dispatcher:  dispatcher,
	}
}

// Root returns the root of tree.
func (s *State) Root() core.Root {
	return s.root
}

// Events returns the registry mapping events to callbacks.
func (s *State) Events() *events.Registry {
	return s.events
}

// PathLog returns the list of active paths.
func (s *State) PathLog() *pathlog.PathLog {
	return s.activePaths
}

// EventDispatcher returns the dispatcher to send events to the experiment.
func (s *State) EventDispatcher() EventDispatcher {
	return s.dispatcher
}

// Reset the state of Multiscope.
func (s *State) Reset() *State {
	return s.factory.NewState()
}

// TreeID returns the ID of the tree to which the state is associated to.
func (s *State) TreeID() core.TreeID {
	return s.factory.TreeID()
}
