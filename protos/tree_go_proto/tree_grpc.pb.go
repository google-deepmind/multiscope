// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.2.0
// - protoc             v3.19.4
// source: tree.proto

package tree_go_proto

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.32.0 or later.
const _ = grpc.SupportPackageIsVersion7

// TreeClient is the client API for Tree service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type TreeClient interface {
	// Browse the structure of the graph.
	GetNodeStruct(ctx context.Context, in *NodeStructRequest, opts ...grpc.CallOption) (*NodeStructReply, error)
	// Request data from nodes in the graph.
	GetNodeData(ctx context.Context, in *NodeDataRequest, opts ...grpc.CallOption) (*NodeDataReply, error)
	// Send events to the backend.
	SendEvents(ctx context.Context, in *SendEventsRequest, opts ...grpc.CallOption) (*SendEventsReply, error)
	// Reset the state of the server including the full tree as well as the events
	// registry.
	ResetState(ctx context.Context, in *ResetStateRequest, opts ...grpc.CallOption) (*ResetStateReply, error)
	// Returns the list of paths for which the data needs to be written if
	// possible.
	ActivePaths(ctx context.Context, in *ActivePathsRequest, opts ...grpc.CallOption) (Tree_ActivePathsClient, error)
	// Request a stream of events from the backend.
	StreamEvents(ctx context.Context, in *StreamEventsRequest, opts ...grpc.CallOption) (Tree_StreamEventsClient, error)
}

type treeClient struct {
	cc grpc.ClientConnInterface
}

func NewTreeClient(cc grpc.ClientConnInterface) TreeClient {
	return &treeClient{cc}
}

func (c *treeClient) GetNodeStruct(ctx context.Context, in *NodeStructRequest, opts ...grpc.CallOption) (*NodeStructReply, error) {
	out := new(NodeStructReply)
	err := c.cc.Invoke(ctx, "/multiscope.Tree/GetNodeStruct", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *treeClient) GetNodeData(ctx context.Context, in *NodeDataRequest, opts ...grpc.CallOption) (*NodeDataReply, error) {
	out := new(NodeDataReply)
	err := c.cc.Invoke(ctx, "/multiscope.Tree/GetNodeData", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *treeClient) SendEvents(ctx context.Context, in *SendEventsRequest, opts ...grpc.CallOption) (*SendEventsReply, error) {
	out := new(SendEventsReply)
	err := c.cc.Invoke(ctx, "/multiscope.Tree/SendEvents", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *treeClient) ResetState(ctx context.Context, in *ResetStateRequest, opts ...grpc.CallOption) (*ResetStateReply, error) {
	out := new(ResetStateReply)
	err := c.cc.Invoke(ctx, "/multiscope.Tree/ResetState", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *treeClient) ActivePaths(ctx context.Context, in *ActivePathsRequest, opts ...grpc.CallOption) (Tree_ActivePathsClient, error) {
	stream, err := c.cc.NewStream(ctx, &Tree_ServiceDesc.Streams[0], "/multiscope.Tree/ActivePaths", opts...)
	if err != nil {
		return nil, err
	}
	x := &treeActivePathsClient{stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

type Tree_ActivePathsClient interface {
	Recv() (*ActivePathsReply, error)
	grpc.ClientStream
}

type treeActivePathsClient struct {
	grpc.ClientStream
}

func (x *treeActivePathsClient) Recv() (*ActivePathsReply, error) {
	m := new(ActivePathsReply)
	if err := x.ClientStream.RecvMsg(m); err != nil {
		return nil, err
	}
	return m, nil
}

func (c *treeClient) StreamEvents(ctx context.Context, in *StreamEventsRequest, opts ...grpc.CallOption) (Tree_StreamEventsClient, error) {
	stream, err := c.cc.NewStream(ctx, &Tree_ServiceDesc.Streams[1], "/multiscope.Tree/StreamEvents", opts...)
	if err != nil {
		return nil, err
	}
	x := &treeStreamEventsClient{stream}
	if err := x.ClientStream.SendMsg(in); err != nil {
		return nil, err
	}
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	return x, nil
}

type Tree_StreamEventsClient interface {
	Recv() (*Event, error)
	grpc.ClientStream
}

type treeStreamEventsClient struct {
	grpc.ClientStream
}

func (x *treeStreamEventsClient) Recv() (*Event, error) {
	m := new(Event)
	if err := x.ClientStream.RecvMsg(m); err != nil {
		return nil, err
	}
	return m, nil
}

// TreeServer is the server API for Tree service.
// All implementations must embed UnimplementedTreeServer
// for forward compatibility
type TreeServer interface {
	// Browse the structure of the graph.
	GetNodeStruct(context.Context, *NodeStructRequest) (*NodeStructReply, error)
	// Request data from nodes in the graph.
	GetNodeData(context.Context, *NodeDataRequest) (*NodeDataReply, error)
	// Send events to the backend.
	SendEvents(context.Context, *SendEventsRequest) (*SendEventsReply, error)
	// Reset the state of the server including the full tree as well as the events
	// registry.
	ResetState(context.Context, *ResetStateRequest) (*ResetStateReply, error)
	// Returns the list of paths for which the data needs to be written if
	// possible.
	ActivePaths(*ActivePathsRequest, Tree_ActivePathsServer) error
	// Request a stream of events from the backend.
	StreamEvents(*StreamEventsRequest, Tree_StreamEventsServer) error
	mustEmbedUnimplementedTreeServer()
}

// UnimplementedTreeServer must be embedded to have forward compatible implementations.
type UnimplementedTreeServer struct {
}

func (UnimplementedTreeServer) GetNodeStruct(context.Context, *NodeStructRequest) (*NodeStructReply, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetNodeStruct not implemented")
}
func (UnimplementedTreeServer) GetNodeData(context.Context, *NodeDataRequest) (*NodeDataReply, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetNodeData not implemented")
}
func (UnimplementedTreeServer) SendEvents(context.Context, *SendEventsRequest) (*SendEventsReply, error) {
	return nil, status.Errorf(codes.Unimplemented, "method SendEvents not implemented")
}
func (UnimplementedTreeServer) ResetState(context.Context, *ResetStateRequest) (*ResetStateReply, error) {
	return nil, status.Errorf(codes.Unimplemented, "method ResetState not implemented")
}
func (UnimplementedTreeServer) ActivePaths(*ActivePathsRequest, Tree_ActivePathsServer) error {
	return status.Errorf(codes.Unimplemented, "method ActivePaths not implemented")
}
func (UnimplementedTreeServer) StreamEvents(*StreamEventsRequest, Tree_StreamEventsServer) error {
	return status.Errorf(codes.Unimplemented, "method StreamEvents not implemented")
}
func (UnimplementedTreeServer) mustEmbedUnimplementedTreeServer() {}

// UnsafeTreeServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to TreeServer will
// result in compilation errors.
type UnsafeTreeServer interface {
	mustEmbedUnimplementedTreeServer()
}

func RegisterTreeServer(s grpc.ServiceRegistrar, srv TreeServer) {
	s.RegisterService(&Tree_ServiceDesc, srv)
}

func _Tree_GetNodeStruct_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NodeStructRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TreeServer).GetNodeStruct(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/multiscope.Tree/GetNodeStruct",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TreeServer).GetNodeStruct(ctx, req.(*NodeStructRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Tree_GetNodeData_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NodeDataRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TreeServer).GetNodeData(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/multiscope.Tree/GetNodeData",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TreeServer).GetNodeData(ctx, req.(*NodeDataRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Tree_SendEvents_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(SendEventsRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TreeServer).SendEvents(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/multiscope.Tree/SendEvents",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TreeServer).SendEvents(ctx, req.(*SendEventsRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Tree_ResetState_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ResetStateRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TreeServer).ResetState(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/multiscope.Tree/ResetState",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TreeServer).ResetState(ctx, req.(*ResetStateRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Tree_ActivePaths_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(ActivePathsRequest)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(TreeServer).ActivePaths(m, &treeActivePathsServer{stream})
}

type Tree_ActivePathsServer interface {
	Send(*ActivePathsReply) error
	grpc.ServerStream
}

type treeActivePathsServer struct {
	grpc.ServerStream
}

func (x *treeActivePathsServer) Send(m *ActivePathsReply) error {
	return x.ServerStream.SendMsg(m)
}

func _Tree_StreamEvents_Handler(srv interface{}, stream grpc.ServerStream) error {
	m := new(StreamEventsRequest)
	if err := stream.RecvMsg(m); err != nil {
		return err
	}
	return srv.(TreeServer).StreamEvents(m, &treeStreamEventsServer{stream})
}

type Tree_StreamEventsServer interface {
	Send(*Event) error
	grpc.ServerStream
}

type treeStreamEventsServer struct {
	grpc.ServerStream
}

func (x *treeStreamEventsServer) Send(m *Event) error {
	return x.ServerStream.SendMsg(m)
}

// Tree_ServiceDesc is the grpc.ServiceDesc for Tree service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var Tree_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "multiscope.Tree",
	HandlerType: (*TreeServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "GetNodeStruct",
			Handler:    _Tree_GetNodeStruct_Handler,
		},
		{
			MethodName: "GetNodeData",
			Handler:    _Tree_GetNodeData_Handler,
		},
		{
			MethodName: "SendEvents",
			Handler:    _Tree_SendEvents_Handler,
		},
		{
			MethodName: "ResetState",
			Handler:    _Tree_ResetState_Handler,
		},
	},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "ActivePaths",
			Handler:       _Tree_ActivePaths_Handler,
			ServerStreams: true,
		},
		{
			StreamName:    "StreamEvents",
			Handler:       _Tree_StreamEvents_Handler,
			ServerStreams: true,
		},
	},
	Metadata: "tree.proto",
}
