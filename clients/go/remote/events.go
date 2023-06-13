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

package remote

import (
	"context"

	"log"
	"multiscope/internal/server/events"
	pb "multiscope/protos/tree_go_proto"
	pbgrpc "multiscope/protos/tree_go_proto"

	"github.com/pkg/errors"
)

type (
	// Callback function.
	Callback = events.Callback

	// Callbacker is able to return a callback.
	Callbacker interface {
		Callback() Callback
	}

	// Events manages client events by receiving events from the backend and dispatching them locally.
	Events struct {
		reg *events.Registry
	}
)

func newEvents(clt *Client) (*Events, error) {
	req := &pb.StreamEventsRequest{TreeId: clt.TreeID()}
	ctx := context.Background()
	strm, err := clt.client.StreamEvents(ctx, req)
	if err != nil {
		return nil, errors.Errorf("cannot stream events from the server: %v", err)
	}
	e := &Events{reg: events.NewRegistry()}
	ackEvent, err := strm.Recv()
	if err != nil {
		return nil, errors.Errorf("cannot receive acknowledgement event from the server: %v", err)
	}
	if ackEvent.Payload != nil {
		return nil, errors.Errorf("incorrect fist event from the server")
	}
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
