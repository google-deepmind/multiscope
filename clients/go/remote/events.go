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
	reg *events.Registry
}

func newEvents(clt pbgrpc.TreeClient) (*Events, error) {
	req := &pb.StreamEventsRequest{}
	ctx := context.Background()
	strm, err := clt.StreamEvents(ctx, req)
	if err != nil {
		return nil, err
	}
	e := &Events{reg: events.NewRegistry()}
	go e.processEvents(strm)
	return e, nil
}

func (e *Events) processEvents(strm pbgrpc.Tree_StreamEventsClient) {
	for {
		event, err := strm.Recv()
		if err != nil {
			log.Printf("cannot receive the next event: %v", err)
			continue
		}
		e.reg.Process(event)
	}
}
