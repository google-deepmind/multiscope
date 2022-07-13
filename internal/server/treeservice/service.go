// Package treeservice implements a GRPC server on top of a stream graph.
package treeservice

import (
	"context"
	"fmt"
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

// TreeServer implements a GRPC server on top of a stream graph.
type TreeServer struct {
	pbgrpc.UnimplementedTreeServer

	state      State
	stateMutex sync.Mutex
	Registry   *Registry

	toActivePaths *apTreeToChan
}

type apTreeToChan struct {
	m   map[pbgrpc.Tree_ActivePathsServer]chan bool
	mut sync.Mutex
}

func (a *apTreeToChan) add(s pbgrpc.Tree_ActivePathsServer) chan bool {
	a.mut.Lock()
	defer a.mut.Unlock()
	a.m[s] = make(chan bool)
	return a.m[s]
}

func (a *apTreeToChan) del(s pbgrpc.Tree_ActivePathsServer) {
	a.mut.Lock()
	defer a.mut.Unlock()
	delete(a.m, s)
}

func (a *apTreeToChan) dispatch() {
	for _, c := range a.m {
		c <- true
	}
}

var (
	_ pbgrpc.TreeServer = (*TreeServer)(nil)
	_ EventDispatcher   = (*TreeServer)(nil)
)

// New returns a service given a server state.
func New(writerRegistry *Registry, state State) *TreeServer {
	return &TreeServer{
		state:         state,
		Registry:      writerRegistry,
		toActivePaths: &apTreeToChan{m: make(map[pbgrpc.Tree_ActivePathsServer]chan bool)},
	}
}

// State returns the current state of the server.
func (s *TreeServer) State() State {
	s.stateMutex.Lock()
	defer s.stateMutex.Unlock()
	return s.state
}

func (s *TreeServer) stateServer() stateServer {
	s.stateMutex.Lock()
	defer s.stateMutex.Unlock()
	return stateServer{state: s.state, writerRegistry: s.Registry}
}

// GetNodeStruct browses the structure of the graph.
func (s *TreeServer) GetNodeStruct(ctx context.Context, req *pb.NodeStructRequest) (*pb.NodeStructReply, error) {
	return s.stateServer().getNodeStruct(ctx, req)
}

// GetNodeData requests data from nodes in the graph.
func (s *TreeServer) GetNodeData(ctx context.Context, req *pb.NodeDataRequest) (*pb.NodeDataReply, error) {
	return s.stateServer().getNodeData(ctx, req)
}

// Dispatch events using the SendEvents entry point.
func (s *TreeServer) Dispatch(path *core.Path, msg proto.Message) error {
	ctx := context.Background()
	any, err := anypb.New(msg)
	if err != nil {
		return err
	}
	reply, err := s.SendEvents(ctx, &pb.SendEventsRequest{
		Events: []*pb.Event{
			&pb.Event{
				Path:    path.PB(),
				Payload: any,
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
		err = multierr.Append(err, fmt.Errorf(errI))
	}
	return err
}

// SendEvents request data from nodes in the graph.
func (s *TreeServer) SendEvents(ctx context.Context, req *pb.SendEventsRequest) (*pb.SendEventsReply, error) {
	return s.stateServer().sendEvents(ctx, req)
}

// StreamEvents using a continuous gRPC stream for the given path.
func (s *TreeServer) StreamEvents(req *pb.StreamEventsRequest, server pbgrpc.Tree_StreamEventsServer) error {
	return s.stateServer().streamEvents(req, server)
}

// ActivePaths streams active paths when the list is modified.
func (s *TreeServer) ActivePaths(req *pb.ActivePathsRequest, srv pbgrpc.Tree_ActivePathsServer) error {
	fromReset := s.toActivePaths.add(srv)
	defer s.toActivePaths.del(srv)
	return s.activePaths(req, srv, fromReset)
}

// activePaths streams active paths when the list is modified.
func (s *TreeServer) activePaths(req *pb.ActivePathsRequest, srv pbgrpc.Tree_ActivePathsServer, fromReset chan bool) error {
	pathLog := s.stateServer().state.PathLog()
	ch := pathLog.Subscribe()
	defer pathLog.Unsubscribe(ch)
	pathLog.DispatchCurrent()

	ctx := srv.Context()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case rep := <-ch:
			if err := srv.Send(rep); err != nil {
				return err
			}
		case <-fromReset:
			pathLog.Unsubscribe(ch)
			pathLog = s.stateServer().state.PathLog()
			ch = pathLog.Subscribe()
			pathLog.DispatchCurrent()
		}
	}
}

// ResetState resets the state of the server.
func (s *TreeServer) ResetState(ctx context.Context, req *pb.ResetStateRequest) (*pb.ResetStateReply, error) {
	s.stateMutex.Lock()
	defer s.stateMutex.Unlock()
	s.state = s.state.Reset()
	s.Registry = s.Registry.ReplaceState(s.state)
	s.toActivePaths.dispatch()
	return &pb.ResetStateReply{}, nil
}

// Desc returns a description of the service.
func (s *TreeServer) Desc() httpgrpc.Registerer {
	return func(srv *grpc.Server) {
		pbgrpc.RegisterTreeServer(srv, s)
	}
}

// RegisterServices registers the main stream service to the server as well as all the services provided by the nodes.
func (s *TreeServer) RegisterServices(grpcServer *grpc.Server) {
	pbgrpc.RegisterTreeServer(grpcServer, s)
	if s.Registry == nil {
		return
	}
	for _, service := range s.Registry.Services() {
		service(grpcServer, s.State)
	}
}
