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
	state treeservice.IDToState
}

var _ pbgrpc.TickersServer = (*Service)(nil)

// RegisterService registers the TensorWriters service to a gRPC server.
func RegisterService(srv grpc.ServiceRegistrar, state treeservice.IDToState) {
	pbgrpc.RegisterTickersServer(srv, &Service{state: state})
}

// NewPlayer creates a new player in the tree.
func (srv *Service) NewPlayer(ctx context.Context, req *pb.NewPlayerRequest) (*pb.NewPlayerResponse, error) {
	state, err := srv.state.State(treeservice.TreeID(req)) // use state throughout this RPC lifetime.
	if err != nil {
		return nil, err
	}
	player := NewPlayer(req.IgnorePause)
	playerPath, err := player.addToTree(state, req.GetPath())
	if err != nil {
		return nil, err
	}
	return &pb.NewPlayerResponse{
		Player: &pb.Player{
			TreeId: req.TreeId,
			Path:   playerPath.PB(),
		},
	}, nil
}

// StoreFrame goes through all the children of a tree to store their state into some kind of storage.
func (srv *Service) StoreFrame(ctx context.Context, req *pb.StoreFrameRequest) (*pb.StoreFrameResponse, error) {
	state, err := srv.state.State(treeservice.TreeID(req.Player)) // use state throughout this RPC lifetime.
	if err != nil {
		return nil, err
	}
	var player *Player
	if err := core.Set(&player, state.Root(), req.GetPlayer()); err != nil {
		desc := fmt.Sprintf("cannot get the player from the tree: %v", err)
		return nil, status.New(codes.InvalidArgument, desc).Err()
	}
	if err := player.StoreFrame(req.GetData()); err != nil {
		return nil, err
	}
	return &pb.StoreFrameResponse{}, nil
}

// NewTicker creates a new ticker in the tree.
func (srv *Service) NewTicker(ctx context.Context, req *pb.NewTickerRequest) (*pb.NewTickerResponse, error) {
	state, err := srv.state.State(treeservice.TreeID(req)) // use state throughout this RPC lifetime.
	if err != nil {
		return nil, err
	}
	ticker := NewTicker()
	tickerPath, err := ticker.addToTree(state, req.GetPath())
	if err != nil {
		return nil, err
	}
	return &pb.NewTickerResponse{
		Ticker: &pb.Ticker{
			TreeId: req.TreeId,
			Path:   tickerPath.PB(),
		},
	}, nil
}

// WriteTicker the data of a ticker.
func (srv *Service) WriteTicker(ctx context.Context, req *pb.WriteTickerRequest) (*pb.WriteTickerResponse, error) {
	state, err := srv.state.State(treeservice.TreeID(req.Ticker)) // use state throughout this RPC lifetime.
	if err != nil {
		return nil, err
	}
	var ticker *Ticker
	if err := core.Set(&ticker, state.Root(), req.GetTicker()); err != nil {
		desc := fmt.Sprintf("cannot get the ticker from the tree: %v", err)
		return nil, status.New(codes.InvalidArgument, desc).Err()
	}
	if err := ticker.Write(req.GetData()); err != nil {
		return nil, err
	}
	return &pb.WriteTickerResponse{}, nil
}
