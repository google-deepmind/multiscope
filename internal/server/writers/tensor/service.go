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

// Package tensor displays Tensors in Multiscope.
package tensor

import (
	"context"
	"fmt"

	"multiscope/internal/server/core"
	"multiscope/internal/server/treeservice"
	pb "multiscope/protos/tensor_go_proto"
	pbgrpc "multiscope/protos/tensor_go_proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Service implements the Writers service.
type Service struct {
	pbgrpc.UnimplementedTensorsServer
	state treeservice.IDToState
}

var _ pbgrpc.TensorsServer = (*Service)(nil)

// RegisterService registers the Writers service to a gRPC server.
func RegisterService(srv grpc.ServiceRegistrar, state treeservice.IDToState) {
	pbgrpc.RegisterTensorsServer(srv, &Service{state: state})
}

// NewWriter creates a new writer for tensor in the tree.
func (srv *Service) NewWriter(ctx context.Context, req *pb.NewWriterRequest) (*pb.NewWriterResponse, error) {
	state := srv.state.State(treeservice.TreeID(req)) // use state throughout this RPC lifetime.
	writer, err := NewWriter()
	if err != nil {
		desc := fmt.Sprintf("cannot create a new Writer: %v", err)
		return nil, status.New(codes.InvalidArgument, desc).Err()
	}
	writerPath, err := writer.addToTree(state, req.GetPath())
	if err != nil {
		return nil, err
	}
	return &pb.NewWriterResponse{
		Writer: &pb.Writer{
			Path: writerPath.PB(),
		},
		DefaultPanelPath: writerPath.PathTo(NodeNameImage).PB(),
	}, nil
}

// ResetWriter resets a tensor writer in the tree.
func (srv *Service) ResetWriter(ctx context.Context, req *pb.ResetWriterRequest) (*pb.ResetWriterResponse, error) {
	state := srv.state.State(treeservice.TreeID(req.Writer)) // use state throughout this RPC lifetime.
	var writer *Writer
	if err := core.Set(&writer, state.Root(), req.GetWriter()); err != nil {
		desc := fmt.Sprintf("cannot get the Writer from the tree: %v", err)
		return nil, status.New(codes.InvalidArgument, desc).Err()
	}
	if err := writer.reset(); err != nil {
		desc := fmt.Sprintf("cannot reset the Writer: %v", err)
		return nil, status.New(codes.InvalidArgument, desc).Err()
	}
	return &pb.ResetWriterResponse{}, nil
}

// WriteTensor writes data to a given node in the tree.
func (srv *Service) Write(ctx context.Context, req *pb.WriteRequest) (rep *pb.WriteResponse, err error) {
	state := srv.state.State(treeservice.TreeID(req.Writer)) // use state throughout this RPC lifetime.
	var writer *Writer
	if err := core.Set(&writer, state.Root(), req.GetWriter()); err != nil {
		desc := fmt.Sprintf("cannot get the Writer from the tree: %v", err)
		return nil, status.New(codes.InvalidArgument, desc).Err()
	}
	if err = writer.write(req.GetTensor()); err != nil {
		desc := fmt.Sprintf("cannot write tensor: %v", err)
		return nil, status.New(codes.InvalidArgument, desc).Err()
	}
	return &pb.WriteResponse{}, nil
}
