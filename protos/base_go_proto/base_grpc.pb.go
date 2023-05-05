// Protocol buffers & services for creating and using basic writers and groups.

// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.3.0
// - protoc             v3.21.12
// source: base.proto

package base_go_proto

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

const (
	BaseWriters_NewGroup_FullMethodName       = "/multiscope.base.BaseWriters/NewGroup"
	BaseWriters_NewProtoWriter_FullMethodName = "/multiscope.base.BaseWriters/NewProtoWriter"
	BaseWriters_WriteProto_FullMethodName     = "/multiscope.base.BaseWriters/WriteProto"
	BaseWriters_NewRawWriter_FullMethodName   = "/multiscope.base.BaseWriters/NewRawWriter"
	BaseWriters_WriteRaw_FullMethodName       = "/multiscope.base.BaseWriters/WriteRaw"
)

// BaseWritersClient is the client API for BaseWriters service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type BaseWritersClient interface {
	// Create a new group in Multiscope.
	NewGroup(ctx context.Context, in *NewGroupRequest, opts ...grpc.CallOption) (*NewGroupResponse, error)
	// Create a new proto writer node in Multiscope.
	NewProtoWriter(ctx context.Context, in *NewProtoWriterRequest, opts ...grpc.CallOption) (*NewProtoWriterResponse, error)
	// Write proto data to Multiscope.
	WriteProto(ctx context.Context, in *WriteProtoRequest, opts ...grpc.CallOption) (*WriteProtoResponse, error)
	// Create a new raw writer node in Multiscope.
	NewRawWriter(ctx context.Context, in *NewRawWriterRequest, opts ...grpc.CallOption) (*NewRawWriterResponse, error)
	// Write raw data to Multiscope.
	WriteRaw(ctx context.Context, in *WriteRawRequest, opts ...grpc.CallOption) (*WriteRawResponse, error)
}

type baseWritersClient struct {
	cc grpc.ClientConnInterface
}

func NewBaseWritersClient(cc grpc.ClientConnInterface) BaseWritersClient {
	return &baseWritersClient{cc}
}

func (c *baseWritersClient) NewGroup(ctx context.Context, in *NewGroupRequest, opts ...grpc.CallOption) (*NewGroupResponse, error) {
	out := new(NewGroupResponse)
	err := c.cc.Invoke(ctx, BaseWriters_NewGroup_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *baseWritersClient) NewProtoWriter(ctx context.Context, in *NewProtoWriterRequest, opts ...grpc.CallOption) (*NewProtoWriterResponse, error) {
	out := new(NewProtoWriterResponse)
	err := c.cc.Invoke(ctx, BaseWriters_NewProtoWriter_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *baseWritersClient) WriteProto(ctx context.Context, in *WriteProtoRequest, opts ...grpc.CallOption) (*WriteProtoResponse, error) {
	out := new(WriteProtoResponse)
	err := c.cc.Invoke(ctx, BaseWriters_WriteProto_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *baseWritersClient) NewRawWriter(ctx context.Context, in *NewRawWriterRequest, opts ...grpc.CallOption) (*NewRawWriterResponse, error) {
	out := new(NewRawWriterResponse)
	err := c.cc.Invoke(ctx, BaseWriters_NewRawWriter_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *baseWritersClient) WriteRaw(ctx context.Context, in *WriteRawRequest, opts ...grpc.CallOption) (*WriteRawResponse, error) {
	out := new(WriteRawResponse)
	err := c.cc.Invoke(ctx, BaseWriters_WriteRaw_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// BaseWritersServer is the server API for BaseWriters service.
// All implementations must embed UnimplementedBaseWritersServer
// for forward compatibility
type BaseWritersServer interface {
	// Create a new group in Multiscope.
	NewGroup(context.Context, *NewGroupRequest) (*NewGroupResponse, error)
	// Create a new proto writer node in Multiscope.
	NewProtoWriter(context.Context, *NewProtoWriterRequest) (*NewProtoWriterResponse, error)
	// Write proto data to Multiscope.
	WriteProto(context.Context, *WriteProtoRequest) (*WriteProtoResponse, error)
	// Create a new raw writer node in Multiscope.
	NewRawWriter(context.Context, *NewRawWriterRequest) (*NewRawWriterResponse, error)
	// Write raw data to Multiscope.
	WriteRaw(context.Context, *WriteRawRequest) (*WriteRawResponse, error)
	mustEmbedUnimplementedBaseWritersServer()
}

// UnimplementedBaseWritersServer must be embedded to have forward compatible implementations.
type UnimplementedBaseWritersServer struct {
}

func (UnimplementedBaseWritersServer) NewGroup(context.Context, *NewGroupRequest) (*NewGroupResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method NewGroup not implemented")
}
func (UnimplementedBaseWritersServer) NewProtoWriter(context.Context, *NewProtoWriterRequest) (*NewProtoWriterResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method NewProtoWriter not implemented")
}
func (UnimplementedBaseWritersServer) WriteProto(context.Context, *WriteProtoRequest) (*WriteProtoResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method WriteProto not implemented")
}
func (UnimplementedBaseWritersServer) NewRawWriter(context.Context, *NewRawWriterRequest) (*NewRawWriterResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method NewRawWriter not implemented")
}
func (UnimplementedBaseWritersServer) WriteRaw(context.Context, *WriteRawRequest) (*WriteRawResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method WriteRaw not implemented")
}
func (UnimplementedBaseWritersServer) mustEmbedUnimplementedBaseWritersServer() {}

// UnsafeBaseWritersServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to BaseWritersServer will
// result in compilation errors.
type UnsafeBaseWritersServer interface {
	mustEmbedUnimplementedBaseWritersServer()
}

func RegisterBaseWritersServer(s grpc.ServiceRegistrar, srv BaseWritersServer) {
	s.RegisterService(&BaseWriters_ServiceDesc, srv)
}

func _BaseWriters_NewGroup_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NewGroupRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(BaseWritersServer).NewGroup(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: BaseWriters_NewGroup_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(BaseWritersServer).NewGroup(ctx, req.(*NewGroupRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _BaseWriters_NewProtoWriter_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NewProtoWriterRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(BaseWritersServer).NewProtoWriter(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: BaseWriters_NewProtoWriter_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(BaseWritersServer).NewProtoWriter(ctx, req.(*NewProtoWriterRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _BaseWriters_WriteProto_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(WriteProtoRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(BaseWritersServer).WriteProto(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: BaseWriters_WriteProto_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(BaseWritersServer).WriteProto(ctx, req.(*WriteProtoRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _BaseWriters_NewRawWriter_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NewRawWriterRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(BaseWritersServer).NewRawWriter(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: BaseWriters_NewRawWriter_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(BaseWritersServer).NewRawWriter(ctx, req.(*NewRawWriterRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _BaseWriters_WriteRaw_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(WriteRawRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(BaseWritersServer).WriteRaw(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: BaseWriters_WriteRaw_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(BaseWritersServer).WriteRaw(ctx, req.(*WriteRawRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// BaseWriters_ServiceDesc is the grpc.ServiceDesc for BaseWriters service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var BaseWriters_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "multiscope.base.BaseWriters",
	HandlerType: (*BaseWritersServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "NewGroup",
			Handler:    _BaseWriters_NewGroup_Handler,
		},
		{
			MethodName: "NewProtoWriter",
			Handler:    _BaseWriters_NewProtoWriter_Handler,
		},
		{
			MethodName: "WriteProto",
			Handler:    _BaseWriters_WriteProto_Handler,
		},
		{
			MethodName: "NewRawWriter",
			Handler:    _BaseWriters_NewRawWriter_Handler,
		},
		{
			MethodName: "WriteRaw",
			Handler:    _BaseWriters_WriteRaw_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "base.proto",
}
