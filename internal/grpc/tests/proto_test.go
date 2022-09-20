package stream_test

import (
	"context"
	"testing"

	"google.golang.org/protobuf/types/known/timestamppb"

	"multiscope/internal/grpc/client"
	"multiscope/internal/grpc/grpctesting"
	"multiscope/internal/server/root"
	"multiscope/internal/server/writers/base"
)

const protoWriterName = "test_proto_writer"

func TestProtoWriter(t *testing.T) {
	rootNode := root.NewRoot()
	// Setup the connection.
	state := grpctesting.NewState(rootNode, nil, nil)
	conn, clt, err := grpctesting.SetupTest(state)
	if err != nil {
		t.Fatal(err)
		return
	}
	defer conn.Close()
	ctx := context.Background()

	// Adding a node to the tree.
	w := base.NewProtoWriter(nil)
	rootNode.AddChild(protoWriterName, w)

	want := &timestamppb.Timestamp{Seconds: 42}
	if err := w.Write(want); err != nil {
		t.Fatalf("error while writing data: %v", err)
	}

	// Query the server and check the data.
	nodes, err := client.PathToNodes(ctx, clt, []string{protoWriterName})
	if err != nil {
		t.Fatal(err)
	}
	data, err := client.NodesData(ctx, clt, nodes)
	if err != nil {
		t.Fatal(err)
	}
	const tickWant = 2
	if data[0].GetTick() != tickWant {
		t.Errorf("Tick label incorrect: got %d, want %d", data[0].GetTick(), tickWant)
	}
	got := &timestamppb.Timestamp{}
	if err := client.ToProto(data[0], got); err != nil {
		t.Fatal(err)
	}
	if got.Seconds != want.Seconds {
		t.Errorf("wrong message: got %q, want %q", got.Seconds, want.Seconds)
	}
}
