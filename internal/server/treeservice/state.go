package treeservice

import (
	"multiscope/internal/server/core"
	"multiscope/internal/server/events"
	"multiscope/internal/server/pathlog"

	"google.golang.org/protobuf/proto"
)

type (
	// StateFactory returns a new state.
	StateFactory interface {
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
func NewState(root core.Root, evnts *events.Registry, activePaths *pathlog.PathLog, dispatcher EventDispatcher) *State {
	return &State{
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
