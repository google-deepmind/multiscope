package remote

import (
	"context"

	"log"
	"multiscope/internal/server/events"
	pb "multiscope/protos/tree_go_proto"
	pbgrpc "multiscope/protos/tree_go_proto"
)

// Events manages client events by receiving events from the backend and dispatching them locally.
type Events struct {
	clt pbgrpc.TreeClient
}

var _ events.Registerer = (*Events)(nil)

// Register a callback to a path.
func (e *Events) Register(p []string, cbF func() events.Callback) (events.Callback, error) {
	cb := cbF()
	pth := Path(p)
	req := &pb.StreamEventsRequest{
		Path: pth.NodePath(),
	}
	ctx := context.Background()
	strm, err := e.clt.StreamEvents(ctx, req)
	if err != nil {
		return nil, err
	}
	go e.processEvents(pth, cb, strm)
	return cb, nil
}

func (e *Events) processEvents(pth Path, cb events.Callback, strm pbgrpc.Tree_StreamEventsClient) {
	for {
		event, err := strm.Recv()
		if err != nil {
			log.Printf("cannot receive the next event for path %q: %v", pth, err)
			continue
		}
		if _, err := cb.Process(event); err != nil {
			log.Printf("event callback for path %q returned error: %v", pth, err)
			continue
		}
	}
}
