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
	clt, err := grpctesting.SetupTest(state)
	if err != nil {
		t.Fatal(err)
		return
	}
	defer clt.Conn().Close()
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
