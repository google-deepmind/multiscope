package treeservice

import (
	"context"
	"fmt"
	"sync"

	"multiscope/internal/server/core"
	pb "multiscope/protos/tree_go_proto"
	pbgrpc "multiscope/protos/tree_go_proto"
)

type (
	protectedState struct {
		protected *State
		mut       sync.Mutex
	}

	// stateServer is an ephemeral structure created to server a request given an instance of state that is guaranteed not to change during the time of the request.
	stateServer struct {
		state protectedState

		toActivePaths *apTreeToChan
	}

	apTreeToChan struct {
		m   map[pbgrpc.Tree_ActivePathsServer]chan bool
		mut sync.Mutex
	}
)

func (ps *protectedState) state() *State {
	ps.mut.Lock()
	defer ps.mut.Unlock()
	return ps.protected
}

func (ps *protectedState) set(state *State) {
	ps.mut.Lock()
	defer ps.mut.Unlock()
	ps.protected = state
}

func (ps *protectedState) reset() {
	ps.mut.Lock()
	defer ps.mut.Unlock()
	ps.protected = ps.protected.Reset()
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

func marshalPBNode(node core.Node, path []string, expandChildren bool) *pb.Node {
	pbNode := &pb.Node{
		Path: &pb.NodePath{Path: path},
		Mime: node.MIME(),
	}
	if len(path) > 0 {
		pbNode.Name = path[len(path)-1]
	}
	parent, ok := node.(core.Parent)
	if !ok {
		return pbNode
	}
	children, err := parent.Children()
	if err != nil {
		pbNode.Error = err.Error()
		return pbNode
	}
	pbNode.HasChildren = len(children) > 0
	if !expandChildren {
		return pbNode
	}
	pbNode.Children = make([]*pb.Node, len(children))
	for i, child := range children {
		childNode, err := parent.Child(child)
		if err != nil {
			pbNode.Children[i] = &pb.Node{
				Error: err.Error(),
			}
			continue
		}
		childPath := append(append([]string{}, path...), child)
		pbNode.Children[i] = marshalPBNode(childNode, childPath, false)
	}
	return pbNode
}

func pathToPBNode(root core.Parent, path []string, expandChildren bool) *pb.Node {
	node, err := core.PathToNode(root, path)
	if err != nil {
		return &pb.Node{
			Error: err.Error(),
			Path:  &pb.NodePath{Path: path},
		}
	}
	return marshalPBNode(node, path, true)
}

func newStateServer(state *State) *stateServer {
	return &stateServer{
		state:         protectedState{protected: state},
		toActivePaths: &apTreeToChan{m: make(map[pbgrpc.Tree_ActivePathsServer]chan bool)},
	}
}

// Browse the structure of the graph.
func (s *stateServer) getNodeStruct(ctx context.Context, req *pb.NodeStructRequest) (*pb.NodeStructReply, error) {
	paths := req.GetPaths()
	if len(paths) == 0 {
		paths = []*pb.NodePath{{}}
	}
	state := s.state.state()
	rep := &pb.NodeStructReply{}
	rep.Nodes = make([]*pb.Node, len(paths))
	root := state.Root()
	for i, path := range paths {
		rep.Nodes[i] = pathToPBNode(root, path.Path, true)
	}
	return rep, nil
}

// Request data from nodes in the graph.
func (s *stateServer) getNodeData(ctx context.Context, req *pb.NodeDataRequest) (*pb.NodeDataReply, error) {
	state := s.state.state()
	state.PathLog().Dispatch(req)
	rep := &pb.NodeDataReply{}
	reqs := req.GetReqs()
	rep.NodeData = make([]*pb.NodeData, len(reqs))
	root := state.Root()
	for i, req := range reqs {
		rep.NodeData[i] = &pb.NodeData{}
		if req == nil {
			continue
		}
		path := req.Path
		if path == nil {
			continue
		}
		root.MarshalData(rep.NodeData[i], path.Path, req.LastTick)
		rep.NodeData[i].Path = &pb.NodePath{Path: path.Path}
	}
	return rep, nil
}

// Send an event to a node in the tree, which will be published to all subscribers.
func (s *stateServer) sendEvents(ctx context.Context, req *pb.SendEventsRequest) (*pb.SendEventsReply, error) {
	rep := &pb.SendEventsReply{
		Errors: make([]string, len(req.Events)),
	}
	state := s.state.state()
	eventRegistry := state.Events()
	if eventRegistry == nil {
		rep.Errors = []string{"Events cannot be processed because the backend does not have an event handler."}
		return rep, nil
	}
	for _, event := range req.Events {
		// Make the sure the path exists and is prefixed with the name of the root node
		// (which the client is not aware of).
		if event.Path == nil {
			event.Path = &pb.NodePath{}
		}
		// Process the event.
		eventRegistry.Process(event)
	}
	return rep, nil
}

// StreamEvents using a continuous gRPC stream for the given path.
func (s *stateServer) streamEvents(req *pb.StreamEventsRequest, server pbgrpc.Tree_StreamEventsServer) error {
	if req.GetPath() == nil {
		return fmt.Errorf("path is required")
	}
	state := s.state.state()
	eventRegistry := state.Events()
	events := eventRegistry.Subscribe(req.GetPath().GetPath(), req.TypeUrl)
	defer eventRegistry.Unsubscribe(events)
	errCh := make(chan error)
	go func() {
		for {
			event, err := events.Next()
			if err != nil {
				errCh <- err
				break
			}
			if event == nil {
				// This is only reported if server.Context().Done() was not closed.
				errCh <- fmt.Errorf("internal error: events.Next() = nil in StreamEvents")
				break
			}
			// TODO(vikrantvarma): this will buffer writes (64kb on the client by
			//  default), and so breaks the last-N semantics of EventQueue. Revisit.
			if err := server.Send(event); err != nil {
				errCh <- err
				break
			}
		}
	}()
	select {
	case <-server.Context().Done():
		return server.Context().Err()
	case err := <-errCh:
		return err
	}
}

// ResetState resets the state of the server.
func (s *stateServer) resetState(ctx context.Context, req *pb.ResetStateRequest) (*pb.ResetStateReply, error) {
	s.state.reset()
	s.toActivePaths.dispatch()
	return &pb.ResetStateReply{}, nil
}

// Delete a node in the tree.
func (s *stateServer) deletePath(ctx context.Context, req *pb.DeleteRequest) (*pb.DeleteReply, error) {
	var path []string
	if req.Path != nil {
		path = req.Path.Path
	}
	state := s.state.state()
	if err := state.Root().Delete(path); err != nil {
		return nil, fmt.Errorf("cannot delete node at path %v: %w", path, err)
	}
	return &pb.DeleteReply{}, nil
}

// ActivePaths streams active paths when the list is modified.
func (s *stateServer) activePaths(req *pb.ActivePathsRequest, srv pbgrpc.Tree_ActivePathsServer) error {
	fromReset := s.toActivePaths.add(srv)
	defer s.toActivePaths.del(srv)
	return s.activePathsWithReset(req, srv, fromReset)
}

func (s *stateServer) activePathsWithReset(req *pb.ActivePathsRequest, srv pbgrpc.Tree_ActivePathsServer, fromReset chan bool) error {
	state := s.state.state()
	pathLog := state.PathLog()
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
			pathLog = state.PathLog()
			ch = pathLog.Subscribe()
			pathLog.DispatchCurrent()
		}
	}
}
