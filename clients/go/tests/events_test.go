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
	"sync"
	"testing"
	"time"

	"multiscope/clients/go/clienttesting"
	"multiscope/clients/go/remote"
	"multiscope/clients/go/scope"
	pb "multiscope/protos/tree_go_proto"

	"github.com/google/go-cmp/cmp"
)

func TestEvents(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	textWriter, err := remote.NewTextWriter(clt, "EventReceiver", nil)
	if err != nil {
		t.Fatal(err)
	}
	received := false
	wg := sync.WaitGroup{}
	wg.Add(1)
	cb := func(ev *scope.Event) error {
		var want []string = textWriter.Path()
		var got = ev.GetPath().GetPath()
		if !cmp.Equal(want, got) {
			t.Errorf("wrong path: got %v but want %v", got, want)
		}
		received = true
		wg.Done()
		return nil
	}
	queue := clt.EventsManager().NewQueueForPath(textWriter.Path(), cb)
	defer queue.Delete()
	// When Register returns, it is not guaranteed the full pipeline between the client
	// and the server is completely setup.
	// Specifically, the client can have a connection but the callback may not be
	// registered to the server.
	time.Sleep(5 * time.Second)
	ctx := context.Background()
	req := &pb.SendEventsRequest{
		Events: []*pb.Event{
			{
				Path: textWriter.Path().NodePath(),
			}},
	}
	_, err = clt.TreeClient().SendEvents(ctx, req)
	if err != nil {
		t.Fatalf("cound not send the event: %v", err)
	}
	wg.Wait()
	if !received {
		t.Error("event has not been received.")
	}
}
