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

	"log"
	"multiscope/internal/grpc/grpctesting"
	"multiscope/internal/server/core"
	"multiscope/internal/server/writers/base"
	pb "multiscope/protos/tree_go_proto"

	"github.com/google/go-cmp/cmp"
)

func addChildTo(nameToPath map[string][]string, root core.Parent, parentPath []string, nodeName string, node core.Node) []string {
	parent, err := core.PathToNode(root, parentPath)
	if err != nil {
		log.Fatal(err)
	}
	parent.(core.ChildAdder).AddChild(nodeName, node)
	path := append(append([]string{}, parentPath...), nodeName)
	nameToPath[nodeName] = path
	return path
}

func buildGraph() (core.Root, map[string][]string) {
	nameToPath := make(map[string][]string)
	root := base.NewRoot()

	g1 := base.NewGroup("")
	g1Path := addChildTo(nameToPath, root, nil, "g1", g1)

	g2 := base.NewGroup("")
	addChildTo(nameToPath, root, nil, "g2", g2)

	g3 := base.NewGroup("")
	g3Path := addChildTo(nameToPath, root, g1Path, "g3", g3)

	g5 := base.NewGroup("")
	addChildTo(nameToPath, root, g3Path, "g5", g5)

	return root, nameToPath
}

func respToNodeMap(rep *pb.NodeStructReply) map[string]*pb.Node {
	nameToNode := make(map[string]*pb.Node)
	for _, node := range rep.GetNodes() {
		for _, child := range node.GetChildren() {
			nameToNode[child.GetName()] = child
		}
	}
	return nameToNode
}

func checkChildren(nameToPath map[string][]string, nameToNode map[string]*pb.Node, children ...string) error {
	if len(nameToNode) != len(children) {
		return fmt.Errorf("node got %d children, want %d children. Children: got %v want %v", len(nameToNode), len(children), nameToNode, children)
	}
	for _, c := range children {
		childNode := nameToNode[c]
		if childNode == nil {
			return fmt.Errorf("%q not found in the children (children: %v)", c, nameToNode)
		}
		want, ok := nameToPath[childNode.GetName()]
		if !ok {
			return fmt.Errorf("id path of child node %q not found", childNode.GetName())
		}
		got := childNode.GetPath().GetPath()
		if diff := cmp.Diff(want, got); diff != "" {
			return fmt.Errorf("invalid id path: got %v want %v diff: %s", got, want, diff)
		}
	}
	return nil
}

func toPaths(nameToNode map[string]*pb.Node, name string) []*pb.NodePath {
	return []*pb.NodePath{nameToNode[name].GetPath()}
}

func TestServerStruct(t *testing.T) {
	root, nameToPath := buildGraph()
	state := grpctesting.NewState(root, nil, nil)
	conn, clt, err := grpctesting.SetupTest(state)
	if err != nil {
		t.Fatal(err)
		return
	}
	defer conn.Close()
	ctx := context.Background()
	// Get the path to root
	rep, err := clt.GetNodeStruct(ctx, &pb.NodeStructRequest{})
	if err != nil {
		t.Errorf("error getting the path to root: %v", err)
	}
	nameToNode := respToNodeMap(rep)
	if err := checkChildren(nameToPath, nameToNode, "g1", "g2"); err != nil {
		t.Fatal(err)
		return
	}
	// Get the children of g1
	rep, err = clt.GetNodeStruct(ctx, &pb.NodeStructRequest{
		Paths: toPaths(nameToNode, "g1"),
	})
	if err != nil {
		t.Fatal(err)
		return
	}
	nameToNode = respToNodeMap(rep)
	if err := checkChildren(nameToPath, nameToNode, "g3"); err != nil {
		t.Fatal(err)
		return
	}
	// Get the children of g3
	rep, err = clt.GetNodeStruct(ctx, &pb.NodeStructRequest{
		Paths: toPaths(nameToNode, "g3"),
	})
	if err != nil {
		t.Fatal(err)
		return
	}
	nameToNode = respToNodeMap(rep)
	if err := checkChildren(nameToPath, nameToNode, "g5"); err != nil {
		t.Fatal(err)
		return
	}
}
