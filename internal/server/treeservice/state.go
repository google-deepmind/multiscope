package treeservice

import (
	"multiscope/internal/server/core"
	"multiscope/internal/server/events"
	"multiscope/internal/server/pathlog"

	"google.golang.org/protobuf/proto"
)

// EventDispatcher dispatches events to the experiment.
type EventDispatcher interface {
	// Dispatch sends an event to the experiment.
	Dispatch(path *core.Path, ev proto.Message) error
}

// State is the state served by the gRPC server.
type State interface {
	// Reset resets the state of the server.
	Reset() State
	// Root returns the root of tree.
	Root() core.Root
	// Events returns the registry mapping events to callbacks.
	Events() *events.Registry
	// PathLog returns the list of active paths.
	PathLog() *pathlog.PathLog
	// EventDispatcher returns the dispatcher to send events to the experiment.
	EventDispatcher() EventDispatcher
}

// Shared is a State shared across all clients.
type Shared struct {
	root        core.Root
	events      *events.Registry
	activePaths *pathlog.PathLog
	dispatcher  EventDispatcher
}

// NewShared returns a server state shared across all clients.
func NewShared(root core.Root, evnts *events.Registry, activePaths *pathlog.PathLog) *Shared {
	return &Shared{
		root:        root,
		events:      evnts,
		activePaths: activePaths,
	}
}

// SetDispatcher sets the dispatcher in the state.
func (sh *Shared) SetDispatcher(dispatcher EventDispatcher) {
	sh.dispatcher = dispatcher
}

// Root returns the root of tree.
func (sh *Shared) Root() core.Root {
	return sh.root
}

// Events returns the registry mapping events to callbacks.
func (sh *Shared) Events() *events.Registry {
	return sh.events
}

// PathLog returns the list of active paths.
func (sh *Shared) PathLog() *pathlog.PathLog {
	return sh.activePaths
}

// EventDispatcher returns the dispatcher to send events to the experiment.
func (sh *Shared) EventDispatcher() EventDispatcher {
	return sh.dispatcher
}
