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
