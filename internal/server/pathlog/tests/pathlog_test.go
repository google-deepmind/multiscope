// Copyright 2023 Google LLC
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

package stats_test

import (
	"context"
	"fmt"
	"testing"
	"time"

	"multiscope/internal/grpc/client"
	"multiscope/internal/grpc/grpctesting"
	"multiscope/internal/server/core"
	"multiscope/internal/server/pathlog"
	"multiscope/internal/server/writers/base"
	pb "multiscope/protos/tree_go_proto"
	pbgrpc "multiscope/protos/tree_go_proto"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/grpc"
)

func checkActiveFromGRPC(clt pbgrpc.Tree_ActivePathsClient, want []core.Key) error {
	rep, err := clt.Recv()
	if err != nil {
		return fmt.Errorf("cannot receive active path message: %v", err)
	}
	got := []core.Key{}
	for _, path := range rep.GetPaths() {
		got = append(got, core.ToKey(path.GetPath()))
	}
	if diff := cmp.Diff(got, want); diff != "" {
		return fmt.Errorf("wrong active paths: got %v want %v diff %v", got, want, diff)
	}
	return nil
}

func checkActiveFromState(pl *pathlog.PathLog, want []core.Key) error {
	for _, key := range want {
		if !pl.IsActive(key) {
			return fmt.Errorf("key %q has not been registered as active", key)
		}
	}
	return nil
}

func checkActive(clt pbgrpc.Tree_ActivePathsClient, pl *pathlog.PathLog, want []core.Key) error {
	if err := checkActiveFromGRPC(clt, want); err != nil {
		return err
	}
	if err := checkActiveFromState(pl, want); err != nil {
		return err
	}
	return nil
}

const activeThreshold = 2 * time.Second

// Build the tree:
// |-A
// |-B
//
//	|-C
//	|-D
func buildTree() core.Root {
	root := base.NewRoot()
	root.AddChild("A", base.NewRawWriter(""))
	grp := base.NewGroup("B")
	root.AddChild("B", grp)
	grp.AddChild("C", base.NewRawWriter(""))
	grp.AddChild("D", base.NewRawWriter(""))
	return root
}

func connectToServer(root core.Root) (conn *grpc.ClientConn, clt pbgrpc.TreeClient, pathLog *pathlog.PathLog, err error) {
	pathLog = pathlog.NewPathLog(activeThreshold)
	state := grpctesting.NewState(root, nil, pathLog)
	conn, clt, err = grpctesting.SetupTest(state)
	return
}

func TestLogPathGRPC(t *testing.T) {
	conn, clt, pathLog, err := connectToServer(buildTree())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	ctx := context.Background()

	// Query the server and check the data.
	nodes, err := client.PathToNodes(ctx, clt, []string{"A"}, []string{"B", "D"})
	if err != nil {
		t.Fatal(err)
	}

	activeClient, err := clt.ActivePaths(ctx, &pb.ActivePathsRequest{})
	if err != nil {
		t.Fatal(err)
	}
	// We check we have an empty list of active nodes when we start.
	if err := checkActive(activeClient, pathLog, []core.Key{}); err != nil {
		t.Error(err)
	}

	// Request A
	pathA := nodes[0]
	if _, err = client.NodesData(ctx, clt, []*pb.Node{pathA}); err != nil {
		t.Fatal(err)
	}
	if err := checkActive(activeClient, pathLog, []core.Key{core.Key("A")}); err != nil {
		t.Error(err)
	}

	// Request B/D so we expect both A and B/D
	pathBD := nodes[1]
	if _, err = client.NodesData(ctx, clt, []*pb.Node{pathBD}); err != nil {
		t.Fatal(err)
	}
	if err := checkActive(activeClient, pathLog, []core.Key{core.Key("A"), core.Key("B/D")}); err != nil {
		t.Error(err)
	}
	// Check that no path are active after the threshold limit.
	time.Sleep(activeThreshold * 2)
	if err := checkActive(activeClient, pathLog, []core.Key{}); err != nil {
		t.Error(err)
	}
}

func toPath(root core.Root, key core.Key) *core.Path {
	return core.NewPath(root).PathTo(key.Split()...)
}

func TestLogPathForward(t *testing.T) {
	root := buildTree()
	conn, clt, pathLog, err := connectToServer(root)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	ctx := context.Background()

	// Add some forwards
	pathLog.Forward(toPath(root, "A"), toPath(root, "B/D"))
	pathLog.Forward(toPath(root, "B"), toPath(root, "A"))
	// Make sure we handle cycle correctly.
	pathLog.Forward(toPath(root, "B/D"), toPath(root, "B"))

	// Query the server and check the data.
	nodes, err := client.PathToNodes(ctx, clt, []string{"A"}, []string{"B", "D"})
	if err != nil {
		t.Fatal(err)
	}

	activeClient, err := clt.ActivePaths(ctx, &pb.ActivePathsRequest{})
	if err != nil {
		t.Fatal(err)
	}
	// We check we have an empty list of active nodes when we start.
	if err := checkActive(activeClient, pathLog, []core.Key{}); err != nil {
		t.Error(err)
	}

	// Request B/D so we expect  A, B, and B/D
	pathBD := nodes[1]
	if _, err = client.NodesData(ctx, clt, []*pb.Node{pathBD}); err != nil {
		t.Fatal(err)
	}
	if err := checkActive(activeClient, pathLog, []core.Key{core.Key("A"), core.Key("B"), core.Key("B/D")}); err != nil {
		t.Error(err)
	}
	// Check that no path are active after the threshold limit.
	time.Sleep(activeThreshold * 2)
	if err := checkActive(activeClient, pathLog, []core.Key{}); err != nil {
		t.Error(err)
	}
}
