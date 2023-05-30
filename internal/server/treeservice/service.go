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

// Package treeservice implements a GRPC server on top of a stream graph.
package treeservice

import (
	"context"
	"errors"
	"sync"

	"multiscope/internal/httpgrpc"
	"multiscope/internal/server/core"
	pb "multiscope/protos/tree_go_proto"
	pbgrpc "multiscope/protos/tree_go_proto"

	"go.uber.org/multierr"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

type (
	// ID is a tree ID.
	ID int64

	// IDToState returns the state of a server and its tree given an ID.
	IDToState interface {
		State(ID) *State
	}

	// RegisterServiceCallback to register a grpc service provided by a node.
	RegisterServiceCallback func(srv grpc.ServiceRegistrar, idToState IDToState)

	// TreeServer implements a GRPC server on top of a stream graph.
	TreeServer struct {
		pbgrpc.UnimplementedTreeServer
		// Services provided by the different node types
		services []RegisterServiceCallback

		idToState  IDToState
		idToServer sync.Map
	}

	// TreeIDGetter gives access to a tree ID (typically a proto).
	TreeIDGetter interface {
		GetTreeID() *pb.TreeID
	}
)

var (
	_ pbgrpc.TreeServer = (*TreeServer)(nil)
	_ EventDispatcher   = (*TreeServer)(nil)
	_ IDToState         = (*TreeServer)(nil)
)

// TreeID returns the ID of a tree given a proto.
func TreeID(msg TreeIDGetter) ID {
	if msg == nil {
		return 0
	}
	treeID := msg.GetTreeID()
	if treeID == nil {
		return 0
	}
	return ID(treeID.TreeID)
}

// New returns a service given a server state.
func New(services []RegisterServiceCallback, idToState IDToState) *TreeServer {
	return &TreeServer{
		services:  services,
		idToState: idToState,
	}
}

func (s *TreeServer) stateServer(id ID) *stateServer {
	if server, ok := s.idToServer.Load(id); ok {
		return server.(*stateServer)
	}
	state := s.idToState.State(id)
	server := newStateServer(state)
	s.idToServer.Store(id, server)
	return server
}

// State returns the current state of the server.
func (s *TreeServer) State(id ID) *State {
	return s.stateServer(id).state.state()
}

// GetNodeStruct browses the structure of the graph.
func (s *TreeServer) GetNodeStruct(ctx context.Context, req *pb.NodeStructRequest) (*pb.NodeStructReply, error) {
	return s.stateServer(TreeID(req)).getNodeStruct(ctx, req)
}

// GetNodeData requests data from nodes in the graph.
func (s *TreeServer) GetNodeData(ctx context.Context, req *pb.NodeDataRequest) (*pb.NodeDataReply, error) {
	return s.stateServer(TreeID(req)).getNodeData(ctx, req)
}

// Dispatch events using the SendEvents entry point.
func (s *TreeServer) Dispatch(path *core.Path, msg proto.Message) error {
	ctx := context.Background()
	pl, err := anypb.New(msg)
	if err != nil {
		return err
	}
	reply, err := s.SendEvents(ctx, &pb.SendEventsRequest{
		Events: []*pb.Event{
			{
				Path:    path.PB(),
				Payload: pl,
			},
		},
	})
	if err != nil {
		return err
	}
	for _, errI := range reply.GetErrors() {
		if errI == "" {
			continue
		}
		err = multierr.Append(err, errors.New(errI))
	}
	return err
}

// SendEvents request data from nodes in the graph.
func (s *TreeServer) SendEvents(ctx context.Context, req *pb.SendEventsRequest) (*pb.SendEventsReply, error) {
	return s.stateServer(TreeID(req)).sendEvents(ctx, req)
}

// StreamEvents using a continuous gRPC stream for the given path.
func (s *TreeServer) StreamEvents(req *pb.StreamEventsRequest, server pbgrpc.Tree_StreamEventsServer) error {
	return s.stateServer(TreeID(req)).streamEvents(req, server)
}

// ActivePaths streams active paths when the list is modified.
func (s *TreeServer) ActivePaths(req *pb.ActivePathsRequest, srv pbgrpc.Tree_ActivePathsServer) error {
	return s.stateServer(TreeID(req)).activePaths(req, srv)
}

// ResetState resets the state of the server.
func (s *TreeServer) ResetState(ctx context.Context, req *pb.ResetStateRequest) (*pb.ResetStateReply, error) {
	return s.stateServer(TreeID(req)).resetState(ctx, req)
}

// Delete a node in the tree.
func (s *TreeServer) Delete(ctx context.Context, req *pb.DeleteRequest) (*pb.DeleteReply, error) {
	return s.stateServer(TreeID(req)).deletePath(ctx, req)
}

// Desc returns a description of the service.
func (s *TreeServer) Desc() httpgrpc.Registerer {
	return func(srv *grpc.Server) {
		pbgrpc.RegisterTreeServer(srv, s)
	}
}

// RegisterServices registers the main stream service to the server as well as all the services provided by the nodes.
func (s *TreeServer) RegisterServices(grpcServer grpc.ServiceRegistrar) {
	pbgrpc.RegisterTreeServer(grpcServer, s)
	for _, service := range s.services {
		service(grpcServer, s)
	}
}
