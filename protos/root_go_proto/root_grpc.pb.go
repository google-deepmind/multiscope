// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.2.0
// - protoc             v3.17.3
// source: root.proto

package root_go_proto

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

// RootClient is the client API for Root service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type RootClient interface {
	// Set the layout of the UI.
	SetLayout(ctx context.Context, in *SetLayoutRequest, opts ...grpc.CallOption) (*SetLayoutResponse, error)
}

type rootClient struct {
	cc grpc.ClientConnInterface
}

func NewRootClient(cc grpc.ClientConnInterface) RootClient {
	return &rootClient{cc}
}

func (c *rootClient) SetLayout(ctx context.Context, in *SetLayoutRequest, opts ...grpc.CallOption) (*SetLayoutResponse, error) {
	out := new(SetLayoutResponse)
	err := c.cc.Invoke(ctx, "/multiscope.root.Root/SetLayout", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// RootServer is the server API for Root service.
// All implementations must embed UnimplementedRootServer
// for forward compatibility
type RootServer interface {
	// Set the layout of the UI.
	SetLayout(context.Context, *SetLayoutRequest) (*SetLayoutResponse, error)
	mustEmbedUnimplementedRootServer()
}

// UnimplementedRootServer must be embedded to have forward compatible implementations.
type UnimplementedRootServer struct {
}

func (UnimplementedRootServer) SetLayout(context.Context, *SetLayoutRequest) (*SetLayoutResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method SetLayout not implemented")
}
func (UnimplementedRootServer) mustEmbedUnimplementedRootServer() {}

// UnsafeRootServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to RootServer will
// result in compilation errors.
type UnsafeRootServer interface {
	mustEmbedUnimplementedRootServer()
}

func RegisterRootServer(s grpc.ServiceRegistrar, srv RootServer) {
	s.RegisterService(&Root_ServiceDesc, srv)
}

func _Root_SetLayout_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(SetLayoutRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RootServer).SetLayout(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/multiscope.root.Root/SetLayout",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RootServer).SetLayout(ctx, req.(*SetLayoutRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// Root_ServiceDesc is the grpc.ServiceDesc for Root service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var Root_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "multiscope.root.Root",
	HandlerType: (*RootServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "SetLayout",
			Handler:    _Root_SetLayout_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "root.proto",
}
