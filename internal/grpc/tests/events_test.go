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
	cbMap := events.NewRegistry()
	state := grpctesting.NewState(root.NewRoot(), cbMap, nil)
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

	cb := events.CallbackF(func(event *pb.Event) (bool, error) {
		wg.Done()
		return true, nil
	})

	if _, err := cbMap.Register(cbPath, func() events.Callback {
		return cb
	}); err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	_, err := client.SendEvents(ctx, req)
	if err != nil {
		t.Error(err)
	}

	wg.Wait()
}
