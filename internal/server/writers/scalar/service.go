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
	state treeservice.StateProvider
}

var _ pbgrpc.ScalarsServer = (*Service)(nil)

// RegisterService registers the TableWriters service to a gRPC server.
func RegisterService(srv grpc.ServiceRegistrar, state treeservice.StateProvider) {
	pbgrpc.RegisterScalarsServer(srv, &Service{state: state})
}

// NewWriter creates a new vega writer in the tree.
func (srv *Service) NewWriter(ctx context.Context, req *pb.NewWriterRequest) (*pb.NewWriterResponse, error) {
	state := srv.state() // use state throughout this RPC lifetime.
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
	state := srv.state() // use state throughout this RPC lifetime.
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
