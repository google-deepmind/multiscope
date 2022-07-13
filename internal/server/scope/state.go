package scope

import (
	"multiscope/internal/server/events"
	"multiscope/internal/server/pathlog"
	"multiscope/internal/server/root"
	"multiscope/internal/server/treeservice"
	"multiscope/internal/server/writers"
	"time"
)

const activePathThreshold = 1 * time.Minute

type serverState struct {
	*treeservice.Shared
}

func newState() *serverState {
	shared := treeservice.NewShared(
		root.NewRoot(),
		events.NewRegistry(),
		pathlog.NewPathLog(activePathThreshold))
	return &serverState{Shared: shared}
}

// Reset the state of Multiscope.
func (s *serverState) Reset() treeservice.State {
	state := newState()
	state.SetDispatcher(s.EventDispatcher())
	return state
}

// NewServer starts a new Multiscope server.
func NewServer() *treeservice.TreeServer {
	state := newState()
	srv := treeservice.New(writers.NewRegistry(state), state)
	state.SetDispatcher(srv)
	return srv
}
