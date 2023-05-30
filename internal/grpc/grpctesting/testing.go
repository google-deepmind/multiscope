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

func newState(root core.Root, eventRegistry *events.Registry, pathLog *pathlog.PathLog) *sharedMock {
	if pathLog == nil {
		pathLog = pathlog.NewPathLog(1 * time.Minute)
	}
	mock := &sharedMock{}
	mock.state = treeservice.NewState(mock, root, eventRegistry, pathLog, nil)
	return mock
}

func (mk *sharedMock) NewState() *treeservice.State {
	mk.mut.Lock()
	defer mk.mut.Unlock()

	return newState(base.NewRoot(), events.NewRegistry(), mk.state.PathLog()).state
}

func (mk *sharedMock) State(id treeservice.ID) *treeservice.State {
	mk.mut.Lock()
	defer mk.mut.Unlock()

	return mk.state
}

// NewState returns a new mock server state.
// If pathLog is nil, a default pathLog is used with one minute as active time.
func NewState(root core.Root, eventRegistry *events.Registry, pathLog *pathlog.PathLog) treeservice.IDToState {
	return newState(root, eventRegistry, pathLog)
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
