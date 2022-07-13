// Package clienttesting provides helper functions to test a Go Multiscope client.
package clienttesting

import (
	"context"
	"fmt"
	"multiscope/clients/go/remote"
	"multiscope/internal/grpc/client"
	"multiscope/internal/server/core"
	"multiscope/internal/server/scope"
	pb "multiscope/protos/tree_go_proto"
	pbgrpc "multiscope/protos/tree_go_proto"
	"sync"
)

// Start a Multiscope gRPC server and returns a client connected to the server.
func Start() (*remote.Client, error) {
	srv := scope.NewServer()
	wg := sync.WaitGroup{}
	grpcPort, err := scope.RunGRPC(srv, &wg, "localhost:0")
	if err != nil {
		return nil, err
	}
	ctx := context.Background()
	// Connect the client.
	return remote.Connect(ctx, fmt.Sprintf("localhost:%d", grpcPort))
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
