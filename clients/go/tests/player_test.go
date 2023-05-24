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

package scope_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"multiscope/clients/go/clienttesting"
	"multiscope/clients/go/remote"
	"multiscope/internal/grpc/client"
	"multiscope/internal/server/writers/ticker/storage"
	"multiscope/internal/server/writers/ticker/tickertesting"
	pb "multiscope/protos/ticker_go_proto"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/testing/protocmp"
)

func TestPlayerTimeline(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}

	// Create the tree.
	player, err := remote.NewPlayer(clt, tickertesting.Ticker01Name, false, nil)
	if err != nil {
		t.Fatal(err)
	}
	writer, err := remote.NewHTMLWriter(clt, tickertesting.Ticker01HTMLWriterName, player.Path())
	if err != nil {
		t.Fatal(err)
	}

	const nTicks = 10
	// Write some data.
	for i := 0; i < nTicks; i++ {
		if err := writer.WriteCSS("css:" + fmt.Sprint(i)); err != nil {
			t.Fatal(err)
		}
		if err := writer.Write("html:" + fmt.Sprint(i)); err != nil {
			t.Fatal(err)
		}
		if err := player.StoreFrame(); err != nil {
			t.Fatal(err)
		}
	}

	if err := tickertesting.CheckPlayerTimeline01(clt.TreeClient(), player.Path(), writer.Path()); err != nil {
		t.Error(err)
	}
	if err := player.Close(); err != nil {
		t.Fatalf("cannot close player: %v", err)
	}
}

func TestPlayerTimelineCleanup(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}

	// Create the tree.
	player, err := remote.NewPlayer(clt, tickertesting.Ticker01Name, false, nil)
	if err != nil {
		t.Fatal(err)
	}

	writer, err := remote.NewTextWriter(clt, "TextWriter", player.Path())
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.Write(""); err != nil {
		t.Fatal(err)
	}

	// Shrink the storage.
	storage.Global().SetSize(5000)

	// Write some data.
	const nTicks = 100
	padding := strings.Repeat("012345678 ", 8)
	for i := uint64(0); i < nTicks; i++ {
		// Each write is 10 bytes.
		if err := writer.Write(fmt.Sprintf("%s  %02d", padding, i)); err != nil {
			t.Fatal(err)
		}
		if err := player.StoreFrame(); err != nil {
			t.Fatal(err)
		}
	}
	// Display the oldest tick.
	if err := tickertesting.SendSetDisplayTick(clt.TreeClient(), player.Path(), 0); err != nil {
		t.Fatal(err)
	}
	// Check the data returned by the timeline.
	// Get the nodes given the path.
	ctx := context.Background()
	nodes, err := client.PathToNodes(ctx, clt.TreeClient(), writer.Path())
	if err != nil {
		t.Fatal(err)
	}
	data, err := client.NodesData(ctx, clt.TreeClient(), nodes)
	if err != nil {
		t.Fatal(err)
	}
	textGot, err := client.ToRaw(data[0])
	if err != nil {
		t.Fatal(err)
	}
	textWant := fmt.Sprintf("%s  %02d", padding, 60)
	if string(textGot) != textWant {
		t.Errorf("got %s, want %s", string(textGot), textWant)
	}
	// Check the timeline display.
	if nodes, err = client.PathToNodes(ctx, clt.TreeClient(), player.Path()); err != nil {
		t.Fatal(err)
	}
	if data, err = client.NodesData(ctx, clt.TreeClient(), nodes); err != nil {
		t.Fatal(err)
	}
	display := &pb.PlayerInfo{}
	if err = client.ToProto(data[0], display); err != nil {
		t.Fatal(err)
	}
	tlGot := display.Timeline
	tlWant := &pb.TimeLine{
		DisplayTick:     60,
		HistoryLength:   40,
		OldestTick:      60,
		StorageCapacity: "5e+03/5e+03 (92%)"}
	if diff := cmp.Diff(tlWant, tlGot, protocmp.Transform()); diff != "" {
		t.Errorf("got the following timeline:\n%v\nbut want the following:\n%v\ndiff:\n%s",
			tlGot, tlWant, diff)
	}
	if err := player.Close(); err != nil {
		t.Fatalf("cannot close player: %v", err)
	}
}

func TestPlayerEmptyTimeline(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}

	// Create the tree.
	player, err := remote.NewPlayer(clt, tickertesting.Ticker01Name, false, nil)
	if err != nil {
		t.Fatal(err)
	}
	writer, err := remote.NewTextWriter(clt, "TextWriter", player.Path())
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	nodes, err := client.PathToNodes(ctx, clt.TreeClient(), writer.Path())
	if err != nil {
		t.Fatal(err)
	}
	// Requesting data when no tick has occurred yet.
	data, err := client.NodesData(ctx, clt.TreeClient(), nodes)
	if err != nil {
		t.Fatal(err)
	}
	const errWant = "data for tick 0 does not exist"
	errGot := data[0].GetError()
	if errGot != errWant {
		t.Errorf("unexpected error for absent data: got %q want %q", errGot, errWant)
	}
	// Requesting data when a tick has occurred, but no data has been written.
	if err := player.StoreFrame(); err != nil {
		t.Fatal(err)
	}
	if data, err = client.NodesData(ctx, clt.TreeClient(), nodes); err != nil {
		t.Fatal(err)
	}
	errGot = data[0].GetError()
	if errGot != "" {
		t.Errorf("unexpected error: got %q want \"\"", errGot)
	}
	if err := player.Close(); err != nil {
		t.Fatalf("cannot close player: %v", err)
	}
}
