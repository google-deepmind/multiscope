// Package grpctesting provides helper functions to test the grpc communication with the stream package.
package grpctesting

import (
	"fmt"
	"sync"
	"time"

	"multiscope/internal/grpc/client"
	"multiscope/internal/server/core"
	"multiscope/internal/server/events"
	"multiscope/internal/server/pathlog"
	"multiscope/internal/server/scope"
	"multiscope/internal/server/treeservice"
	"multiscope/internal/server/writers/base"
	pbgrpc "multiscope/protos/tree_go_proto"

	"google.golang.org/grpc"
)

type sharedMock struct {
	state *treeservice.State
	mut   sync.Mutex
}

func newState(root core.Root, eventRegistry *events.Registry, pathLog *pathlog.PathLog) *treeservice.State {
	return treeservice.NewState(root, eventRegistry, pathLog, nil)
}

func (mk *sharedMock) NewState() *treeservice.State {
	mk.mut.Lock()
	defer mk.mut.Unlock()

	mk.state = newState(base.NewRoot(), events.NewRegistry(), mk.state.PathLog())
	return mk.state
}

func (mk *sharedMock) State(id treeservice.ID) *treeservice.State {
	mk.mut.Lock()
	defer mk.mut.Unlock()

	return mk.state
}

// NewState returns a new mock server state.
// If pathLog is nil, a default pathLog is used with one minute as active time.
func NewState(root core.Root, eventRegistry *events.Registry, pathLog *pathlog.PathLog) treeservice.IDToState {
	if pathLog == nil {
		pathLog = pathlog.NewPathLog(1 * time.Minute)
	}
	return &sharedMock{
		state: newState(root, eventRegistry, pathLog),
	}
}

// SetupTest starts a grpc server, create a connection to that server, and then returns it alongside with a stream client.
func SetupTest(state treeservice.IDToState, services ...treeservice.RegisterServiceCallback) (*grpc.ClientConn, pbgrpc.TreeClient, error) {
	srv := treeservice.New(services, state)
	wg := sync.WaitGroup{}
	addr, err := scope.RunGRPC(srv, &wg, "localhost:0")
	if err != nil {
		return nil, nil, err
	}
	conn, err := client.Connect(fmt.Sprintf("localhost:%d", addr.Port))
	if err != nil {
		return nil, nil, err
	}
	return conn, pbgrpc.NewTreeClient(conn), nil
}
