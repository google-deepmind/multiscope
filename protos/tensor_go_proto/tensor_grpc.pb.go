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

// Protocol buffer to stream tensors.

// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.3.0
// - protoc             v3.21.12
// source: tensor.proto

package tensor_go_proto

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
	Tensors_NewWriter_FullMethodName   = "/multiscope.tensors.Tensors/NewWriter"
	Tensors_ResetWriter_FullMethodName = "/multiscope.tensors.Tensors/ResetWriter"
	Tensors_Write_FullMethodName       = "/multiscope.tensors.Tensors/Write"
)

// TensorsClient is the client API for Tensors service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type TensorsClient interface {
	// Create a new tensor writer node in Multiscope.
	NewWriter(ctx context.Context, in *NewWriterRequest, opts ...grpc.CallOption) (*NewWriterResponse, error)
	// Reset the data of a writer.
	ResetWriter(ctx context.Context, in *ResetWriterRequest, opts ...grpc.CallOption) (*ResetWriterResponse, error)
	// Write tensor data to Multiscope.
	Write(ctx context.Context, in *WriteRequest, opts ...grpc.CallOption) (*WriteResponse, error)
}

type tensorsClient struct {
	cc grpc.ClientConnInterface
}

func NewTensorsClient(cc grpc.ClientConnInterface) TensorsClient {
	return &tensorsClient{cc}
}

func (c *tensorsClient) NewWriter(ctx context.Context, in *NewWriterRequest, opts ...grpc.CallOption) (*NewWriterResponse, error) {
	out := new(NewWriterResponse)
	err := c.cc.Invoke(ctx, Tensors_NewWriter_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *tensorsClient) ResetWriter(ctx context.Context, in *ResetWriterRequest, opts ...grpc.CallOption) (*ResetWriterResponse, error) {
	out := new(ResetWriterResponse)
	err := c.cc.Invoke(ctx, Tensors_ResetWriter_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *tensorsClient) Write(ctx context.Context, in *WriteRequest, opts ...grpc.CallOption) (*WriteResponse, error) {
	out := new(WriteResponse)
	err := c.cc.Invoke(ctx, Tensors_Write_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// TensorsServer is the server API for Tensors service.
// All implementations must embed UnimplementedTensorsServer
// for forward compatibility
type TensorsServer interface {
	// Create a new tensor writer node in Multiscope.
	NewWriter(context.Context, *NewWriterRequest) (*NewWriterResponse, error)
	// Reset the data of a writer.
	ResetWriter(context.Context, *ResetWriterRequest) (*ResetWriterResponse, error)
	// Write tensor data to Multiscope.
	Write(context.Context, *WriteRequest) (*WriteResponse, error)
	mustEmbedUnimplementedTensorsServer()
}

// UnimplementedTensorsServer must be embedded to have forward compatible implementations.
type UnimplementedTensorsServer struct {
}

func (UnimplementedTensorsServer) NewWriter(context.Context, *NewWriterRequest) (*NewWriterResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method NewWriter not implemented")
}
func (UnimplementedTensorsServer) ResetWriter(context.Context, *ResetWriterRequest) (*ResetWriterResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method ResetWriter not implemented")
}
func (UnimplementedTensorsServer) Write(context.Context, *WriteRequest) (*WriteResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Write not implemented")
}
func (UnimplementedTensorsServer) mustEmbedUnimplementedTensorsServer() {}

// UnsafeTensorsServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to TensorsServer will
// result in compilation errors.
type UnsafeTensorsServer interface {
	mustEmbedUnimplementedTensorsServer()
}

func RegisterTensorsServer(s grpc.ServiceRegistrar, srv TensorsServer) {
	s.RegisterService(&Tensors_ServiceDesc, srv)
}

func _Tensors_NewWriter_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NewWriterRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TensorsServer).NewWriter(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Tensors_NewWriter_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TensorsServer).NewWriter(ctx, req.(*NewWriterRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Tensors_ResetWriter_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ResetWriterRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TensorsServer).ResetWriter(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Tensors_ResetWriter_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TensorsServer).ResetWriter(ctx, req.(*ResetWriterRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Tensors_Write_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(WriteRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TensorsServer).Write(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Tensors_Write_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TensorsServer).Write(ctx, req.(*WriteRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// Tensors_ServiceDesc is the grpc.ServiceDesc for Tensors service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var Tensors_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "multiscope.tensors.Tensors",
	HandlerType: (*TensorsServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "NewWriter",
			Handler:    _Tensors_NewWriter_Handler,
		},
		{
			MethodName: "ResetWriter",
			Handler:    _Tensors_ResetWriter_Handler,
		},
		{
			MethodName: "Write",
			Handler:    _Tensors_Write_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "tensor.proto",
}
