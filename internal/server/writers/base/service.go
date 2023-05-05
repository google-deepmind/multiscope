package base

import (
	"context"

	"multiscope/internal/server/core"
	"multiscope/internal/server/treeservice"
	pb "multiscope/protos/base_go_proto"
	pbgrpc "multiscope/protos/base_go_proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Service implements the BaseWriters service.
type Service struct {
	pbgrpc.UnimplementedBaseWritersServer
	state treeservice.IDToState
}

var _ pbgrpc.BaseWritersServer = (*Service)(nil)

// RegisterService registers the BaseWriters service to a gRPC server.
func RegisterService(srv grpc.ServiceRegistrar, state treeservice.IDToState) {
	pbgrpc.RegisterBaseWritersServer(srv, &Service{state: state})
}

// NewGroup creates a new parent node in the tree.
func (srv *Service) NewGroup(ctx context.Context, req *pb.NewGroupRequest) (*pb.NewGroupResponse, error) {
	state := srv.state.State(treeservice.TreeID(req)) // use state throughout this RPC lifetime.
	grp := NewGroup("")
	writerPath, err := grp.AddToTree(state, req.GetPath())
	if err != nil {
		return nil, err
	}
	return &pb.NewGroupResponse{
		Grp: &pb.Group{
			Path: writerPath.PB(),
		},
	}, nil
}

// NewProtoWriter creates a new writer for proto in the tree.
func (srv *Service) NewProtoWriter(ctx context.Context, req *pb.NewProtoWriterRequest) (*pb.NewProtoWriterResponse, error) {
	state := srv.state.State(treeservice.TreeID(req)) // use state throughout this RPC lifetime.
	writer := NewProtoWriter(req.GetProto())
	writerPath, err := writer.AddToTree(state, req.GetPath())
	if err != nil {
		return nil, err
	}
	return &pb.NewProtoWriterResponse{
		Writer: &pb.ProtoWriter{
			Path: writerPath.PB(),
		},
	}, nil
}

// WriteProto writes proto data to a given node in the tree.
func (srv *Service) WriteProto(ctx context.Context, req *pb.WriteProtoRequest) (*pb.WriteProtoResponse, error) {
	state := srv.state.State(treeservice.TreeID(req.Writer)) // use state throughout this RPC lifetime.
	var writer *ProtoWriter
	if err := core.Set(&writer, state.Root(), req.GetWriter()); err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "cannot get the ProtoWriter from the tree: %v", err)
	}
	pr := req.GetProto()
	if pr == nil {
		writer.WriteBytes("", nil)
	} else {
		writer.WriteBytes(pr.GetTypeUrl(), pr.GetValue())
	}
	return &pb.WriteProtoResponse{}, nil
}

// NewRawWriter creates a new writer for raw data in the tree.
func (srv *Service) NewRawWriter(ctx context.Context, req *pb.NewRawWriterRequest) (*pb.NewRawWriterResponse, error) {
	state := srv.state.State(treeservice.TreeID(req)) // use state throughout this RPC lifetime.
	writer := NewRawWriter(req.GetMime())
	writerPath, err := writer.AddToTree(state, req.GetPath())
	if err != nil {
		return nil, err
	}
	return &pb.NewRawWriterResponse{
		Writer: &pb.RawWriter{
			Path: writerPath.PB(),
		},
	}, nil
}

// WriteRaw writes raw data to a given node in the tree.
func (srv *Service) WriteRaw(ctx context.Context, req *pb.WriteRawRequest) (*pb.WriteRawResponse, error) {
	state := srv.state.State(treeservice.TreeID(req.Writer)) // use state throughout this RPC lifetime.
	var writer *RawWriter
	if err := core.Set(&writer, state.Root(), req.GetWriter()); err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "cannot get the RawWriter from the tree: %v", err)
	}
	err := writer.Write(req.GetData())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "cannot write data: %v", err)
	}
	return &pb.WriteRawResponse{}, nil
}
