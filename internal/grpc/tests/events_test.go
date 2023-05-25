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

package stream_test

import (
	"context"
	"sync"
	"testing"

	"multiscope/internal/grpc/grpctesting"
	"multiscope/internal/server/events"
	"multiscope/internal/server/root"
	"multiscope/internal/server/treeservice"

	pb "multiscope/protos/tree_go_proto"
)

func TestServerDispatchesEvents(t *testing.T) {
	reg := events.NewRegistry()
	state := grpctesting.NewState(root.NewRoot(), reg, nil)
	client := treeservice.New(nil, state)

	req := &pb.SendEventsRequest{
		Events: []*pb.Event{
			{
				Path: &pb.NodePath{Path: []string{"node"}},
			},
		},
	}

	cbPath := []string{"node"}
	var wg sync.WaitGroup
	wg.Add(1)
	queue := reg.NewQueueForPath(cbPath, func(event *pb.Event) error {
		wg.Done()
		return nil
	})
	defer queue.Delete()

	ctx := context.Background()
	_, err := client.SendEvents(ctx, req)
	if err != nil {
		t.Error(err)
	}

	wg.Wait()
}
