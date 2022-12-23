package uimain

import (
	"context"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

// toServer sends events to the server.
type toServer struct {
	ui        ui.UI
	toProcess chan *treepb.SendEventsRequest
}

func newToServer(ui ui.UI) *toServer {
	to := &toServer{ui: ui, toProcess: make(chan *treepb.SendEventsRequest, 10)}
	go to.processEvents()
	return to
}

func (to *toServer) processEvents() {
	clt := to.ui.TreeClient()
	for req := range to.toProcess {
		ctx := context.Background()
		if _, err := clt.SendEvents(ctx, req); err != nil {
			to.ui.DisplayErr(err)
		}
	}
}

func (to *toServer) sendEvent(path *treepb.NodePath, msg proto.Message) {
	var event anypb.Any
	if err := anypb.MarshalFrom(&event, msg, proto.MarshalOptions{}); err != nil {
		to.ui.DisplayErr(err)
		return
	}
	to.toProcess <- &treepb.SendEventsRequest{
		Events: []*treepb.Event{{
			Path:    path,
			Payload: &event,
		}},
	}
}
