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
	"fmt"
	"net/url"
	"sync"

	"multiscope/internal/httpgrpc"
	"multiscope/internal/server/core"
	"multiscope/internal/version"
	pb "multiscope/protos/tree_go_proto"
	pbgrpc "multiscope/protos/tree_go_proto"

	"github.com/pkg/errors"
	"go.uber.org/multierr"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

type (
	// URLToState returns a possibly new server state given a URL.
	URLToState interface {
		ToState(url *url.URL) (*State, error)

		Delete(core.TreeID)
	}

	// IDToState returns a state given an ID.
	// An error is returned if no state can be found given the ID.
	IDToState interface {
		State(id core.TreeID) (*State, error)
	}

	// RegisterServiceCallback to register a grpc service provided by a node.
	RegisterServiceCallback func(srv grpc.ServiceRegistrar, idToState IDToState)

	// TreeServer implements a GRPC server on top of a stream graph.
	TreeServer struct {
		pbgrpc.UnimplementedTreeServer
		// Services provided by the different node types
		services []RegisterServiceCallback

		urlToState URLToState
		idToServer sync.Map
	}

	// TreeIDGetter gives access to a tree ID (typically a proto).
	TreeIDGetter interface {
		GetTreeId() *pb.TreeID
	}
)

var (
	_ pbgrpc.TreeServer = (*TreeServer)(nil)
	_ EventDispatcher   = (*TreeServer)(nil)
	_ IDToState         = (*TreeServer)(nil)
)

// TreeID returns the ID of a tree given a proto.
func TreeID(msg TreeIDGetter) core.TreeID {
	if msg == nil {
		return 0
	}
	treeID := msg.GetTreeId()
	if treeID == nil {
		return 0
	}
	return core.TreeID(treeID.TreeId)
}

// New returns a service given a server state.
func New(services []RegisterServiceCallback, urlToState URLToState) *TreeServer {
	return &TreeServer{
		services:   services,
		urlToState: urlToState,
	}
}

func (s *TreeServer) stateServer(id core.TreeID) (*stateServer, bool) {
	serverWithState, ok := s.idToServer.Load(id)
	if !ok {
		return nil, false
	}
	return serverWithState.(*stateServer), true
}

func idErr(id core.TreeID) error {
	return errors.Errorf("ID %v cannot be found", id)
}

// State returns the current state of the server.
func (s *TreeServer) State(id core.TreeID) (*State, error) {
	serverWithState, ok := s.stateServer(id)
	if !ok {
		return nil, idErr(id)
	}
	return serverWithState.state.state(), nil
}

// GetTreeID returns a tree ID from a URL.
func (s *TreeServer) GetTreeID(ctx context.Context, req *pb.GetTreeIDRequest) (*pb.GetTreeIDReply, error) {
	url, err := url.Parse(req.Url)
	if err != nil {
		return nil, errors.Errorf("cannot parse URL %q: %v", req.Url, err)
	}
	state, err := s.urlToState.ToState(url)
	if err != nil {
		return nil, fmt.Errorf("cannot get a tree ID for URL %q: %w", req.Url, err)
	}
	resp := &pb.GetTreeIDReply{
		TreeId: &pb.TreeID{
			TreeId: int64(state.TreeID()),
		},
		Version: version.Version,
	}
	serverWithState, ok := s.stateServer(state.TreeID())
	if ok {
		return resp, nil
	}
	serverWithState = newStateServer(state)
	s.idToServer.Store(state.TreeID(), serverWithState)
	return resp, nil
}

// GetNodeStruct browses the structure of the graph.
func (s *TreeServer) GetNodeStruct(ctx context.Context, req *pb.NodeStructRequest) (*pb.NodeStructReply, error) {
	serverWithState, ok := s.stateServer(TreeID(req))
	if !ok {
		return nil, idErr(TreeID(req))
	}
	return serverWithState.getNodeStruct(ctx, req)
}

// GetNodeData requests data from nodes in the graph.
func (s *TreeServer) GetNodeData(ctx context.Context, req *pb.NodeDataRequest) (*pb.NodeDataReply, error) {
	serverWithState, ok := s.stateServer(TreeID(req))
	if !ok {
		return nil, idErr(TreeID(req))
	}
	return serverWithState.getNodeData(ctx, req)
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
	serverWithState, ok := s.stateServer(TreeID(req))
	if !ok {
		return nil, idErr(TreeID(req))
	}
	return serverWithState.sendEvents(ctx, req)
}

// StreamEvents using a continuous gRPC stream for the given path.
func (s *TreeServer) StreamEvents(req *pb.StreamEventsRequest, server pbgrpc.Tree_StreamEventsServer) error {
	serverWithState, ok := s.stateServer(TreeID(req))
	if !ok {
		return idErr(TreeID(req))
	}
	return serverWithState.streamEvents(req, server)
}

// ActivePaths streams active paths when the list is modified.
func (s *TreeServer) ActivePaths(req *pb.ActivePathsRequest, srv pbgrpc.Tree_ActivePathsServer) error {
	serverWithState, ok := s.stateServer(TreeID(req))
	if !ok {
		return idErr(TreeID(req))
	}
	return serverWithState.activePaths(req, srv)
}

// ResetState resets the state of the server.
func (s *TreeServer) ResetState(ctx context.Context, req *pb.ResetStateRequest) (*pb.ResetStateReply, error) {
	serverWithState, ok := s.stateServer(TreeID(req))
	if !ok {
		return nil, idErr(TreeID(req))
	}
	return serverWithState.resetState(ctx, req)
}

// Delete a node in the tree.
func (s *TreeServer) Delete(ctx context.Context, req *pb.DeleteRequest) (*pb.DeleteReply, error) {
	serverWithState, ok := s.stateServer(TreeID(req))
	if !ok {
		return nil, idErr(TreeID(req))
	}
	return serverWithState.deletePath(ctx, req)
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
