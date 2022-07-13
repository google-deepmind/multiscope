package root

import (
	"context"

	"multiscope/internal/server/treeservice"
	pb "multiscope/protos/root_go_proto"
	pbgrpc "multiscope/protos/root_go_proto"

	"google.golang.org/grpc"
)

// Service implements the Display service.
type Service struct {
	pbgrpc.UnimplementedRootServer
	state treeservice.StateProvider
}

// RegisterService registers the Display service to a gRPC server.
func RegisterService(srv *grpc.Server, state treeservice.StateProvider) {
	pbgrpc.RegisterRootServer(srv, &Service{state: state})
}

// SetLayout sets the UI layout.
func (srv *Service) SetLayout(ctx context.Context, req *pb.SetLayoutRequest) (*pb.SetLayoutResponse, error) {
	state := srv.state() // use state throughout this RPC lifetime.
	root := state.Root().(*Root)
	if err := root.setLayout(req.Layout); err != nil {
		return nil, err
	}
	return &pb.SetLayoutResponse{}, nil
}
