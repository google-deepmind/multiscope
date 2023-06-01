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

package root

import (
	"context"

	"multiscope/internal/server/treeservice"
	"multiscope/internal/version"
	pb "multiscope/protos/root_go_proto"
	pbgrpc "multiscope/protos/root_go_proto"

	"google.golang.org/grpc"
)

// Service implements the Display service.
type Service struct {
	pbgrpc.UnimplementedRootServer
	state treeservice.IDToState
}

var _ pbgrpc.RootServer = (*Service)(nil)

// RegisterService registers the Display service to a gRPC server.
func RegisterService(srv grpc.ServiceRegistrar, state treeservice.IDToState) {
	pbgrpc.RegisterRootServer(srv, &Service{state: state})
}

// GetVersion returns the version of the gRPC API of the server.
func (srv *Service) GetVersion(ctx context.Context, req *pb.GetVersionRequest) (*pb.GetVersionResponse, error) {
	return &pb.GetVersionResponse{Version: version.Version}, nil
}

// SetKeySettings sets a global name to fetch the UI settings.
func (srv *Service) SetKeySettings(ctx context.Context, req *pb.SetKeySettingsRequest) (*pb.SetKeySettingsResponse, error) {
	state, err := srv.state.State(treeservice.TreeID(req)) // use state throughout this RPC lifetime.
	if err != nil {
		return nil, err
	}
	root := state.Root().(*Root)
	if err := root.setKeySettings(req.KeySettings); err != nil {
		return nil, err
	}
	return &pb.SetKeySettingsResponse{}, nil
}

// SetLayout sets the UI layout.
func (srv *Service) SetLayout(ctx context.Context, req *pb.SetLayoutRequest) (*pb.SetLayoutResponse, error) {
	state, err := srv.state.State(treeservice.TreeID(req)) // use state throughout this RPC lifetime.
	if err != nil {
		return nil, err
	}
	root := state.Root().(*Root)
	if err := root.setLayout(req.Layout); err != nil {
		return nil, err
	}
	return &pb.SetLayoutResponse{}, nil
}

// GetRootInfo returns information about the root node.
func (srv *Service) GetRootInfo(ctx context.Context, req *pb.GetRootInfoRequest) (*pb.GetRootInfoResponse, error) {
	state, err := srv.state.State(treeservice.TreeID(req)) // use state throughout this RPC lifetime.
	if err != nil {
		return nil, err
	}
	root := state.Root().(*Root)
	return &pb.GetRootInfoResponse{
		Info: root.cloneInfo(),
	}, nil
}
