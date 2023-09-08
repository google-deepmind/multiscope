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

package uimain

import (
	"context"
	"fmt"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

// toServer sends events to the server.
type toServer struct {
	ui        ui.UI
	toProcess chan *treepb.Event
}

func newToServer(ui ui.UI) *toServer {
	to := &toServer{ui: ui, toProcess: make(chan *treepb.Event, 100)}
	go to.processEvents()
	return to
}

func (to *toServer) pullEvents() []*treepb.Event {
	events := []*treepb.Event{<-to.toProcess}
	gotEvent := true
	for gotEvent {
		select {
		case ev := <-to.toProcess:
			events = append(events, ev)
			gotEvent = true
		default:
			gotEvent = false
		}
		if len(events) > 100 {
			return events
		}
	}
	return events
}

func (to *toServer) processEvents() {
	clt, _ := to.ui.TreeClient()
	for {
		ctx := context.Background()
		_, treeID := to.ui.TreeClient()
		req := &treepb.SendEventsRequest{
			TreeId: treeID,
			Events: to.pullEvents(),
		}
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
	select {
	case to.toProcess <- &treepb.Event{
		Path:    path,
		Payload: &event,
	}:
	default:
		fmt.Printf("Dropping event %T\n", msg)
	}
}
