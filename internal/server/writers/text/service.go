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

package text

import (
	"context"
	"fmt"

	"multiscope/internal/server/core"
	"multiscope/internal/server/treeservice"
	pb "multiscope/protos/text_go_proto"
	pbgrpc "multiscope/protos/text_go_proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Service implements the TablesWriter and Writer service.
type Service struct {
	pbgrpc.UnimplementedTextServer
	state treeservice.IDToState
}

// RegisterService registers the TableWriters service to a gRPC server.
func RegisterService(srv grpc.ServiceRegistrar, state treeservice.IDToState) {
	pbgrpc.RegisterTextServer(srv, &Service{state: state})
}

// NewWriter creates a new raw writer in the tree.
func (srv *Service) NewWriter(ctx context.Context, req *pb.NewWriterRequest) (*pb.NewWriterResponse, error) {
	state, err := srv.state.State(treeservice.TreeID(req)) // use state throughout this RPC lifetime.
	if err != nil {
		return nil, err
	}
	writer := NewWriter()
	writerPath, err := core.SetNodeAt(state.Root(), req.GetPath(), writer)
	if err != nil {
		return nil, err
	}
	return &pb.NewWriterResponse{
		Writer: &pb.Writer{
			TreeId: req.TreeId,
			Path:   writerPath.PB(),
		},
	}, nil
}

// NewHTMLWriter creates a new HTML writer in the tree.
func (srv *Service) NewHTMLWriter(ctx context.Context, req *pb.NewHTMLWriterRequest) (*pb.NewHTMLWriterResponse, error) {
	state, err := srv.state.State(treeservice.TreeID(req)) // use state throughout this RPC lifetime.
	if err != nil {
		return nil, err
	}
	writer := NewHTMLWriter()
	writerPath, err := writer.AddToTree(state, req.GetPath())
	if err != nil {
		return nil, err
	}
	return &pb.NewHTMLWriterResponse{
		Writer: &pb.HTMLWriter{
			TreeId: req.TreeId,
			Path:   writerPath.PB(),
		},
	}, nil
}

// Write writes raw text to a Writer in the tree.
func (srv *Service) Write(ctx context.Context, req *pb.WriteRequest) (rep *pb.WriteResponse, err error) {
	state, err := srv.state.State(treeservice.TreeID(req.Writer)) // use state throughout this RPC lifetime.
	if err != nil {
		return nil, err
	}
	var writer *Writer
	if err := core.Set(&writer, state.Root(), req.GetWriter()); err != nil {
		desc := fmt.Sprintf("cannot get the Writer from the tree: %v", err)
		return nil, status.New(codes.InvalidArgument, desc).Err()
	}
	if err := writer.Write(req.GetText()); err != nil {
		return nil, err
	}
	return &pb.WriteResponse{}, nil
}

// WriteHTML writes HTML text to a HTMLWriter in the tree.
func (srv *Service) WriteHTML(ctx context.Context, req *pb.WriteHTMLRequest) (rep *pb.WriteHTMLResponse, err error) {
	state, err := srv.state.State(treeservice.TreeID(req.Writer)) // use state throughout this RPC lifetime.
	if err != nil {
		return nil, err
	}
	var writer *HTMLWriter
	if err := core.Set(&writer, state.Root(), req.GetWriter()); err != nil {
		desc := fmt.Sprintf("cannot get the HTMLWriter from the tree: %v", err)
		return nil, status.New(codes.InvalidArgument, desc).Err()
	}
	if err := writer.Write(req.GetHtml()); err != nil {
		return nil, err
	}
	return &pb.WriteHTMLResponse{}, nil
}

// WriteCSS writes CSS text to a HTMLWriter in the tree.
func (srv *Service) WriteCSS(ctx context.Context, req *pb.WriteCSSRequest) (rep *pb.WriteCSSResponse, err error) {
	state, err := srv.state.State(treeservice.TreeID(req.Writer)) // use state throughout this RPC lifetime.
	if err != nil {
		return nil, err
	}
	var writer *HTMLWriter
	if err := core.Set(&writer, state.Root(), req.GetWriter()); err != nil {
		desc := fmt.Sprintf("cannot get the HTMLWriter from the tree: %v", err)
		return nil, status.New(codes.InvalidArgument, desc).Err()
	}
	if err := writer.WriteCSS(req.GetCss()); err != nil {
		return nil, err
	}
	return &pb.WriteCSSResponse{}, nil
}
