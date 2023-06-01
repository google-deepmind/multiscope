package grpctesting

import (
	"multiscope/internal/server/core"
	"multiscope/internal/server/events"
	"multiscope/internal/server/pathlog"
	"multiscope/internal/server/treeservice"
	"multiscope/internal/server/writers/base"
	"net/url"
	"sync"
	"time"
)

type sharedMock struct {
	state *treeservice.State
	mut   sync.Mutex
}

func newState(root core.Root, eventRegistry *events.Registry, pathLog *pathlog.PathLog) *sharedMock {
	if pathLog == nil {
		pathLog = pathlog.NewPathLog(1 * time.Minute)
	}
	mock := &sharedMock{}
	mock.state = treeservice.NewState(mock, root, eventRegistry, pathLog, nil)
	return mock
}

func (mk *sharedMock) TreeID() core.TreeID {
	return 1
}

func (mk *sharedMock) ToState(url *url.URL) (*treeservice.State, error) {
	return mk.state, nil
}

func (mk *sharedMock) Delete(core.TreeID) {
}

func (mk *sharedMock) NewState() *treeservice.State {
	mk.mut.Lock()
	defer mk.mut.Unlock()

	return newState(base.NewRoot(), events.NewRegistry(), mk.state.PathLog()).state
}

// NewState returns a new mock server state.
// If pathLog is nil, a default pathLog is used with one minute as active time.
func NewState(root core.Root, eventRegistry *events.Registry, pathLog *pathlog.PathLog) treeservice.URLToState {
	return newState(root, eventRegistry, pathLog)
}
