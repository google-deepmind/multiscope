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

package stream_test

import (
	"context"
	"fmt"
	"testing"

	"multiscope/internal/grpc/grpctesting"
	"multiscope/internal/server/writers/base"
	pb "multiscope/protos/tree_go_proto"
)

func TestServerClient(t *testing.T) {
	root := base.NewRoot()
	state := grpctesting.NewState(root, nil, nil)
	client, err := grpctesting.SetupTest(state)
	if err != nil {
		t.Fatal(err)
		return
	}
	defer client.Conn().Close()
	const nChild = 5
	for i := 0; i < nChild; i++ {
		root.AddChild(fmt.Sprintf("child%d", i), base.NewGroup(""))
	}
	ctx := context.Background()
	structRequest := &pb.NodeStructRequest{TreeId: client.TreeID()}
	reply, err := client.TreeClient().GetNodeStruct(ctx, structRequest)
	if err != nil {
		t.Error(err)
	}
	nodes := reply.GetNodes()
	if len(nodes) != 1 {
		t.Fatalf("wrong length of nodes: got %v, want a list of a single element", nodes)
	}
	children := make(map[string]bool)
	for _, child := range nodes[0].GetChildren() {
		children[child.GetName()] = true
	}
	for i := 0; i < nChild; i++ {
		name := fmt.Sprintf("child%d", i)
		if !children[name] {
			t.Errorf("cannot find child %q in %v", name, children)
		}
	}
}

func TestHasChildren(t *testing.T) {
	root := base.NewRoot()
	state := grpctesting.NewState(root, nil, nil)
	client, err := grpctesting.SetupTest(state)
	if err != nil {
		t.Fatal(err)
		return
	}
	defer client.Conn().Close()

	ctx := context.Background()
	structRequest := &pb.NodeStructRequest{TreeId: client.TreeID()}
	reply, err := client.TreeClient().GetNodeStruct(ctx, structRequest)
	if err != nil {
		t.Error(err)
	}
	if len(reply.GetNodes()) != 1 {
		t.Fatalf("wrong length of nodes: got %v, want a list of a single element", reply.GetNodes())
	}
	if reply.GetNodes()[0].GetHasChildren() {
		t.Fatalf("node.GetHasChildren() = true, want false")
	}
	root.AddChild("", base.NewGroup(""))
	reply, err = client.TreeClient().GetNodeStruct(ctx, structRequest)
	if err != nil {
		t.Error(err)
	}
	if len(reply.GetNodes()) != 1 {
		t.Fatalf("wrong length of nodes: got %v, want a list of a single element", reply.GetNodes())
	}
	if !reply.GetNodes()[0].GetHasChildren() {
		t.Fatalf("node.GetHasChildren() = false, want true")
	}
}
