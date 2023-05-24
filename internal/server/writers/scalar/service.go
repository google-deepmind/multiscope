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

package scalar

import (
	"context"
	"fmt"

	"multiscope/internal/server/core"
	"multiscope/internal/server/treeservice"
	pb "multiscope/protos/scalar_go_proto"
	pbgrpc "multiscope/protos/scalar_go_proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Service implements the TablesWriter and Writer service.
type Service struct {
	pbgrpc.UnimplementedScalarsServer
	state treeservice.IDToState
}

var _ pbgrpc.ScalarsServer = (*Service)(nil)

// RegisterService registers the TableWriters service to a gRPC server.
func RegisterService(srv grpc.ServiceRegistrar, state treeservice.IDToState) {
	pbgrpc.RegisterScalarsServer(srv, &Service{state: state})
}

// NewWriter creates a new vega writer in the tree.
func (srv *Service) NewWriter(ctx context.Context, req *pb.NewWriterRequest) (*pb.NewWriterResponse, error) {
	state := srv.state.State(treeservice.TreeID(req)) // use state throughout this RPC lifetime.
	writer := NewWriter()
	writerPath, err := writer.AddToTree(state, req.GetPath())
	if err != nil {
		return nil, err
	}
	return &pb.NewWriterResponse{
		Writer: &pb.Writer{
			Path: writerPath.PB(),
		},
	}, nil
}

// WriteSpec writes a vega spec in the tree.
func (srv *Service) Write(ctx context.Context, req *pb.WriteRequest) (rep *pb.WriteResponse, err error) {
	state := srv.state.State(treeservice.TreeID(req.Writer)) // use state throughout this RPC lifetime.
	var writer *Writer
	if err := core.Set(&writer, state.Root(), req.GetWriter()); err != nil {
		desc := fmt.Sprintf("cannot get the Writer from the tree: %v", err)
		return nil, status.New(codes.InvalidArgument, desc).Err()
	}
	if err := writer.Write(req.GetLabelToValue()); err != nil {
		return nil, err
	}
	return &pb.WriteResponse{}, nil
}
