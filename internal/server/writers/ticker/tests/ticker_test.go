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

package tickerfast_test

import (
	"context"
	"testing"

	"multiscope/internal/grpc/grpctesting"
	"multiscope/internal/server/root"
	"multiscope/internal/server/writers/ticker"
	"multiscope/internal/server/writers/ticker/tickertesting"
	pb "multiscope/protos/ticker_go_proto"
	pbgrpc "multiscope/protos/ticker_go_proto"
	treepb "multiscope/protos/tree_go_proto"
)

func TestFastTicker(t *testing.T) {
	state := grpctesting.NewState(root.NewRoot(), nil, nil)
	conn, clt, err := grpctesting.SetupTest(state, ticker.RegisterService)
	if err != nil {
		t.Fatal(err)
		return
	}
	defer conn.Close()
	tickerClt := pbgrpc.NewTickersClient(conn)

	ctx := context.Background()
	rep, err := tickerClt.NewTicker(ctx, &pb.NewTickerRequest{
		Path: &treepb.NodePath{
			Path: []string{tickertesting.Ticker01Name},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	writer := rep.GetTicker()
	for i, data := range tickertesting.TickerData {
		if _, err := tickerClt.WriteTicker(ctx, &pb.WriteTickerRequest{
			Ticker: writer,
			Data:   data.Data,
		}); err != nil {
			t.Fatal(err)
		}
		if err := tickertesting.CheckTicker(clt, writer.GetPath().GetPath(), i); err != nil {
			t.Errorf("error checking fast ticker from data %d: %v", i, err)
		}
	}
}
