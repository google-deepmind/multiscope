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
	conn, client, err := grpctesting.SetupTest(state)
	if err != nil {
		t.Fatal(err)
		return
	}
	defer conn.Close()
	const nChild = 5
	for i := 0; i < nChild; i++ {
		root.AddChild(fmt.Sprintf("child%d", i), base.NewGroup(""))
	}
	ctx := context.Background()
	structRequest := &pb.NodeStructRequest{}
	reply, err := client.GetNodeStruct(ctx, structRequest)
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
	conn, client, err := grpctesting.SetupTest(state)
	if err != nil {
		t.Fatal(err)
		return
	}
	defer conn.Close()

	ctx := context.Background()
	structRequest := &pb.NodeStructRequest{}
	reply, err := client.GetNodeStruct(ctx, structRequest)
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
	reply, err = client.GetNodeStruct(ctx, structRequest)
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
