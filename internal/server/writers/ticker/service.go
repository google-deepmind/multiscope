// Package ticker implements a ticker to monitor time steps.
package ticker

import (
	"context"
	"fmt"

	"multiscope/internal/server/core"
	"multiscope/internal/server/treeservice"
	pb "multiscope/protos/ticker_go_proto"
	pbgrpc "multiscope/protos/ticker_go_proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Service implements the Tickers service.
type Service struct {
	pbgrpc.UnimplementedTickersServer
	state treeservice.StateProvider
}

var _ pbgrpc.TickersServer = (*Service)(nil)

// RegisterService registers the TensorWriters service to a gRPC server.
func RegisterService(srv *grpc.Server, state treeservice.StateProvider) {
	pbgrpc.RegisterTickersServer(srv, &Service{state: state})
}

// New creates a new ticker in the tree.
func (srv *Service) New(ctx context.Context, req *pb.NewTickerRequest) (*pb.NewTickerResponse, error) {
	state := srv.state() // use state throughout this RPC lifetime.
	ticker := NewTicker()
	tickerPath, err := ticker.addToTree(state, req.GetPath())
	if err != nil {
		return nil, err
	}
	return &pb.NewTickerResponse{
		Ticker: &pb.Ticker{
			Path: tickerPath.PB(),
		},
	}, nil
}

// Write the data of a ticker.
func (srv *Service) Write(ctx context.Context, req *pb.WriteRequest) (*pb.WriteResponse, error) {
	state := srv.state() // use state throughout this RPC lifetime.
	var ticker *Ticker
	if err := core.Set(&ticker, state.Root(), req.GetTicker()); err != nil {
		desc := fmt.Sprintf("cannot get the ticker from the tree: %v", err)
		return nil, status.New(codes.InvalidArgument, desc).Err()
	}
	if err := ticker.Write(req.GetData()); err != nil {
		return nil, err
	}
	return &pb.WriteResponse{}, nil
}
