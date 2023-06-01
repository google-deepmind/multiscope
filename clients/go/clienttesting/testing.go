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

// Package clienttesting provides helper functions to test a Go Multiscope client.
package clienttesting

import (
	"context"
	"errors"
	"fmt"
	"multiscope/clients/go/remote"
	"multiscope/internal/grpc/client"
	"multiscope/internal/server/core"
	"multiscope/internal/server/scope"
	pb "multiscope/protos/tree_go_proto"
	pbgrpc "multiscope/protos/tree_go_proto"
	"sync"
	"time"

	"go.uber.org/multierr"
)

// Start a Multiscope gRPC server and returns a client connected to the server.
func Start() (*remote.Client, error) {
	srv := scope.NewServer()
	wg := sync.WaitGroup{}
	addr, err := scope.RunGRPC(srv, &wg, "localhost:0")
	if err != nil {
		return nil, err
	}
	ctx := context.Background()
	// Connect the client.
	return remote.Connect(ctx, fmt.Sprintf("localhost:%d", addr.Port))
}

// ForceActive forces a path and its children to be active by requesting data from them.
func ForceActive(clt pbgrpc.TreeClient, paths ...[]string) error {
	var err error
	var nodes []*pb.Node
	ctx := context.Background()
	done := make(map[core.Key]bool)
	allDone := false
	// Expand the paths to include all children.
	for !allDone {
		nodes, err = client.PathToNodes(ctx, clt, paths...)
		if err != nil {
			return err
		}
		allDone = true
		for _, node := range nodes {
			nodePath := node.GetPath().GetPath()
			key := core.ToKey(nodePath)
			if done[key] {
				continue
			}
			done[key] = true
			for _, child := range node.GetChildren() {
				childPath := append([]string{}, nodePath...)
				childPath = append(childPath, child.GetName())
				paths = append(paths, childPath)
			}
			allDone = false
		}
	}
	// Query their data twice which should be enough to force the paths to be active.
	if _, err := client.NodesData(ctx, clt, nodes); err != nil {
		return err
	}
	if _, err := client.NodesData(ctx, clt, nodes); err != nil {
		return err
	}
	return nil
}

func checkPathToNodesResponse(nodes []*pb.Node) error {
	var err error
	for _, node := range nodes {
		errS := node.GetError()
		if len(errS) > 0 {
			err = multierr.Append(err, errors.New(errS))
		}
	}
	return err
}

// CheckBecomeActive checks if a writer becomes active after one of its child has been queried.
func CheckBecomeActive(clt *remote.Client, childPath []string, writer remote.WithShouldWrite) error {
	// Check that the writer is not active.
	if writer.ShouldWrite() {
		return fmt.Errorf("%T.ShouldWrite should be equal to false", writer)
	}
	// Query one of the child.
	ctx := context.Background()
	nodes, err := client.PathToNodes(ctx, clt.TreeClient(), childPath)
	if err != nil {
		return err
	}
	if err := checkPathToNodesResponse(nodes); err != nil {
		return err
	}
	if _, err := client.NodesData(ctx, clt.TreeClient(), nodes); err != nil {
		return err
	}
	// Check that the writer becomes active.
	now := time.Now()
	const timeout = 5 * time.Second
	var shouldWrite bool
	for !shouldWrite && time.Since(now) < timeout {
		shouldWrite = writer.ShouldWrite()
	}
	if !shouldWrite {
		return fmt.Errorf("%T.ShouldWrite was not equal to true after %v", writer, timeout)
	}
	return nil
}
