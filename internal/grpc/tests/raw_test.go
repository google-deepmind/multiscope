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
	"time"

	"multiscope/internal/grpc/client"
	"multiscope/internal/grpc/grpctesting"
	"multiscope/internal/server/root"
	"multiscope/internal/server/writers/base"

	"github.com/google/go-cmp/cmp"
)

const rawWriterName = "test_raw_writer"

func TestRawWriter(t *testing.T) {
	rootNode := root.NewRoot()
	state := grpctesting.NewState(rootNode, nil, nil)
	conn, clt, err := grpctesting.SetupTest(state)
	if err != nil {
		t.Fatal(err)
		return
	}
	defer conn.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Adding a node to the tree.
	w := base.NewRawWriter("")
	rootNode.AddChild(rawWriterName, w)

	want := []byte{1, 2, 3, 4}
	if err := w.Write(want); err != nil {
		t.Fatalf("error while writing data: %v", err)
	}

	// Query the server and check the data.
	nodes, err := client.PathToNodes(ctx, clt, []string{rawWriterName})
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
	got, err := client.ToRaw(data[0])
	if err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(got, want); len(diff) > 0 {
		t.Errorf("bytes do not match:\n%s", diff)
	}
}
