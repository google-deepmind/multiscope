package root

import (
	"context"

	"multiscope/internal/server/treeservice"
	"multiscope/protos"
	pb "multiscope/protos/root_go_proto"
	pbgrpc "multiscope/protos/root_go_proto"

	"google.golang.org/grpc"
)

// Service implements the Display service.
type Service struct {
	pbgrpc.UnimplementedRootServer
	state treeservice.StateProvider
}

var _ pbgrpc.RootServer = (*Service)(nil)

// RegisterService registers the Display service to a gRPC server.
func RegisterService(srv grpc.ServiceRegistrar, state treeservice.StateProvider) {
	pbgrpc.RegisterRootServer(srv, &Service{state: state})
}

// GetVersion returns the version of the gRPC API of the server.
func (srv *Service) GetVersion(ctx context.Context, req *pb.GetVersionRequest) (*pb.GetVersionResponse, error) {
	return &pb.GetVersionResponse{Version: protos.Version}, nil
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
