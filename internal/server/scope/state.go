package scope

import (
	"multiscope/internal/server/events"
	"multiscope/internal/server/pathlog"
	"multiscope/internal/server/root"
	"multiscope/internal/server/treeservice"
	"multiscope/internal/server/writers"
	"sync"
	"time"
)

const activePathThreshold = 1 * time.Minute

type singleton struct {
	state *treeservice.State
	mut   sync.Mutex

	dispatcher treeservice.EventDispatcher
}

var _ treeservice.IDToState = (*singleton)(nil)

func (s *singleton) newState() {
	s.state = treeservice.NewState(
		root.NewRoot(),
		events.NewRegistry(),
		pathlog.NewPathLog(activePathThreshold),
		s.dispatcher)
}

func (s *singleton) NewState() *treeservice.State {
	s.mut.Lock()
	defer s.mut.Unlock()
	s.newState()
	return s.state
}

func (s *singleton) State(id treeservice.ID) *treeservice.State {
	s.mut.Lock()
	defer s.mut.Unlock()
	if s.state == nil {
		s.newState()
	}
	return s.state
}

// NewServer starts a new Multiscope server.
func NewServer() *treeservice.TreeServer {
	state := &singleton{}
	srv := treeservice.New(writers.All(), state)
	state.dispatcher = srv
	return srv
}
