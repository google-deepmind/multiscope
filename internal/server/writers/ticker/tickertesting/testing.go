// Package tickertesting provides helper function to test tickers.
package tickertesting

import (
	"context"
	"fmt"
	"multiscope/internal/grpc/client"
	"multiscope/internal/mime"
	tickerpb "multiscope/protos/ticker_go_proto"
	pbgrpc "multiscope/protos/tree_go_proto"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protocmp"
	"google.golang.org/protobuf/types/known/durationpb"
)

// Ticker01Name is the name of the ticker node.
const Ticker01Name = "Ticker01"

var (
	// TickerData is the data to send to a ticker to test it.
	TickerData = []struct {
		Tick int64
		Data *tickerpb.TickerData
	}{
		{
			Data: &tickerpb.TickerData{
				Periods: &tickerpb.TickerData_Periods{
					Total:      durationpb.New(12345 * time.Second),
					Experiment: durationpb.New(12345 * time.Second),
					Callbacks:  durationpb.New(12345 * time.Second),
					Idle:       durationpb.New(12345 * time.Second),
				},
				Tick: 10,
			},
		},
		{
			Data: &tickerpb.TickerData{
				Periods: &tickerpb.TickerData_Periods{
					Total:      durationpb.New(12346 * time.Second),
					Experiment: durationpb.New(12346 * time.Second),
					Callbacks:  durationpb.New(12346 * time.Second),
					Idle:       durationpb.New(12346 * time.Second),
				},
				Tick: 11,
			},
		},
	}

	wantMime = mime.NamedProtobuf(string(proto.MessageName(&tickerpb.TickerData{})))
)

// CheckTicker checks the data written to a fast ticker.
func CheckTicker(clt pbgrpc.TreeClient, path []string, i int) error {
	ctx := context.Background()
	path = append([]string{}, path...)
	nodes, err := client.PathToNodes(ctx, clt, path)
	if err != nil {
		return err
	}
	if diff := cmp.Diff(nodes[0].GetMime(), wantMime); len(diff) > 0 {
		return fmt.Errorf("mime type error: %s", diff)
	}
	data, err := client.NodesData(ctx, clt, nodes)
	if err != nil {
		return err
	}
	got := &tickerpb.TickerData{}
	if err = client.ToProto(data[0], got); err != nil {
		return err
	}
	if diff := cmp.Diff(got, TickerData[i].Data, protocmp.Transform()); len(diff) > 0 {
		return fmt.Errorf("time series do not match:\n%s", diff)
	}
	return nil
}

// CheckRemoteTicker checks the data written to a fast ticker.
func CheckRemoteTicker(clt pbgrpc.TreeClient, path []string, currentTick int64) error {
	ctx := context.Background()
	path = append([]string{}, path...)
	nodes, err := client.PathToNodes(ctx, clt, path)
	if err != nil {
		return err
	}
	if diff := cmp.Diff(nodes[0].GetMime(), wantMime); len(diff) > 0 {
		return fmt.Errorf("mime type error: %s", diff)
	}

	data, err := client.NodesData(ctx, clt, nodes)
	if err != nil {
		return err
	}
	got := tickerpb.TickerData{}
	if err := client.ToProto(data[0], &got); err != nil {
		return err
	}
	if got.Tick != currentTick {
		return fmt.Errorf("got the wrong tick number: got %d want %d", got.Tick, currentTick)
	}
	return nil
}
