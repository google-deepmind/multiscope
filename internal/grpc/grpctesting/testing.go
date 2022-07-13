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
	*treeservice.Shared
}

func (mk *sharedMock) Reset() treeservice.State {
	return NewState(base.NewRoot(), events.NewRegistry(), mk.PathLog())
}

// NewState returns a new mock server state.
// If pathLog is nil, a default pathLog is used with one minute as active time.
func NewState(root core.Root, eventRegistry *events.Registry, pathLog *pathlog.PathLog) treeservice.State {
	if pathLog == nil {
		pathLog = pathlog.NewPathLog(1 * time.Minute)
	}
	return &sharedMock{
		Shared: treeservice.NewShared(root, eventRegistry, pathLog),
	}
}

// SetupTest starts a grpc server, create a connection to that server, and then returns it alongside with a stream client.
func SetupTest(state treeservice.State, services ...treeservice.RegisterServiceCallback) (*grpc.ClientConn, pbgrpc.TreeClient, error) {
	registry := treeservice.NewRegistry(state)
	for _, service := range services {
		registry.RegisterService(service)
	}
	srv := treeservice.New(registry, state)
	wg := sync.WaitGroup{}
	port, err := scope.RunGRPC(srv, &wg, "localhost:0")
	if err != nil {
		return nil, nil, err
	}
	conn, err := client.Connect(fmt.Sprintf("localhost:%d", port))
	if err != nil {
		return nil, nil, err
	}
	return conn, pbgrpc.NewTreeClient(conn), nil

}
