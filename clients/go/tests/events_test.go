package scope_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"multiscope/clients/go/clienttesting"
	"multiscope/clients/go/remote"
	"multiscope/clients/go/scope"
	"multiscope/internal/server/events"
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
	cb := events.CallbackF(func(ev *scope.Event) (bool, error) {
		var want []string = textWriter.Path()
		var got = ev.GetPath().GetPath()
		if !cmp.Equal(want, got) {
			t.Errorf("wrong path: got %v but want %v", got, want)
		}
		received = true
		wg.Done()
		return true, nil
	})
	if _, err = clt.EventsManager().Register(textWriter.Path(), func() events.Callback {
		return cb
	}); err != nil {
		t.Fatal(err)
	}
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
