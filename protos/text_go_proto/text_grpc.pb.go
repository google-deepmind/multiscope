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

// Protocol buffer to stream text data.

// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.3.0
// - protoc             v3.21.12
// source: text.proto

package text_go_proto

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
	Text_NewWriter_FullMethodName     = "/multiscope.text.Text/NewWriter"
	Text_NewHTMLWriter_FullMethodName = "/multiscope.text.Text/NewHTMLWriter"
	Text_Write_FullMethodName         = "/multiscope.text.Text/Write"
	Text_WriteHTML_FullMethodName     = "/multiscope.text.Text/WriteHTML"
	Text_WriteCSS_FullMethodName      = "/multiscope.text.Text/WriteCSS"
)

// TextClient is the client API for Text service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type TextClient interface {
	// Create a new raw writer node in Multiscope.
	NewWriter(ctx context.Context, in *NewWriterRequest, opts ...grpc.CallOption) (*NewWriterResponse, error)
	// Create a new HTML writer node in Multiscope.
	NewHTMLWriter(ctx context.Context, in *NewHTMLWriterRequest, opts ...grpc.CallOption) (*NewHTMLWriterResponse, error)
	// Write raw text to Multiscope.
	Write(ctx context.Context, in *WriteRequest, opts ...grpc.CallOption) (*WriteResponse, error)
	// Write HTML text to Multiscope.
	WriteHTML(ctx context.Context, in *WriteHTMLRequest, opts ...grpc.CallOption) (*WriteHTMLResponse, error)
	// Write CSS text to Multiscope.
	WriteCSS(ctx context.Context, in *WriteCSSRequest, opts ...grpc.CallOption) (*WriteCSSResponse, error)
}

type textClient struct {
	cc grpc.ClientConnInterface
}

func NewTextClient(cc grpc.ClientConnInterface) TextClient {
	return &textClient{cc}
}

func (c *textClient) NewWriter(ctx context.Context, in *NewWriterRequest, opts ...grpc.CallOption) (*NewWriterResponse, error) {
	out := new(NewWriterResponse)
	err := c.cc.Invoke(ctx, Text_NewWriter_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *textClient) NewHTMLWriter(ctx context.Context, in *NewHTMLWriterRequest, opts ...grpc.CallOption) (*NewHTMLWriterResponse, error) {
	out := new(NewHTMLWriterResponse)
	err := c.cc.Invoke(ctx, Text_NewHTMLWriter_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *textClient) Write(ctx context.Context, in *WriteRequest, opts ...grpc.CallOption) (*WriteResponse, error) {
	out := new(WriteResponse)
	err := c.cc.Invoke(ctx, Text_Write_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *textClient) WriteHTML(ctx context.Context, in *WriteHTMLRequest, opts ...grpc.CallOption) (*WriteHTMLResponse, error) {
	out := new(WriteHTMLResponse)
	err := c.cc.Invoke(ctx, Text_WriteHTML_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *textClient) WriteCSS(ctx context.Context, in *WriteCSSRequest, opts ...grpc.CallOption) (*WriteCSSResponse, error) {
	out := new(WriteCSSResponse)
	err := c.cc.Invoke(ctx, Text_WriteCSS_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// TextServer is the server API for Text service.
// All implementations must embed UnimplementedTextServer
// for forward compatibility
type TextServer interface {
	// Create a new raw writer node in Multiscope.
	NewWriter(context.Context, *NewWriterRequest) (*NewWriterResponse, error)
	// Create a new HTML writer node in Multiscope.
	NewHTMLWriter(context.Context, *NewHTMLWriterRequest) (*NewHTMLWriterResponse, error)
	// Write raw text to Multiscope.
	Write(context.Context, *WriteRequest) (*WriteResponse, error)
	// Write HTML text to Multiscope.
	WriteHTML(context.Context, *WriteHTMLRequest) (*WriteHTMLResponse, error)
	// Write CSS text to Multiscope.
	WriteCSS(context.Context, *WriteCSSRequest) (*WriteCSSResponse, error)
	mustEmbedUnimplementedTextServer()
}

// UnimplementedTextServer must be embedded to have forward compatible implementations.
type UnimplementedTextServer struct {
}

func (UnimplementedTextServer) NewWriter(context.Context, *NewWriterRequest) (*NewWriterResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method NewWriter not implemented")
}
func (UnimplementedTextServer) NewHTMLWriter(context.Context, *NewHTMLWriterRequest) (*NewHTMLWriterResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method NewHTMLWriter not implemented")
}
func (UnimplementedTextServer) Write(context.Context, *WriteRequest) (*WriteResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Write not implemented")
}
func (UnimplementedTextServer) WriteHTML(context.Context, *WriteHTMLRequest) (*WriteHTMLResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method WriteHTML not implemented")
}
func (UnimplementedTextServer) WriteCSS(context.Context, *WriteCSSRequest) (*WriteCSSResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method WriteCSS not implemented")
}
func (UnimplementedTextServer) mustEmbedUnimplementedTextServer() {}

// UnsafeTextServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to TextServer will
// result in compilation errors.
type UnsafeTextServer interface {
	mustEmbedUnimplementedTextServer()
}

func RegisterTextServer(s grpc.ServiceRegistrar, srv TextServer) {
	s.RegisterService(&Text_ServiceDesc, srv)
}

func _Text_NewWriter_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NewWriterRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TextServer).NewWriter(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Text_NewWriter_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TextServer).NewWriter(ctx, req.(*NewWriterRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Text_NewHTMLWriter_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NewHTMLWriterRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TextServer).NewHTMLWriter(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Text_NewHTMLWriter_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TextServer).NewHTMLWriter(ctx, req.(*NewHTMLWriterRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Text_Write_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(WriteRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TextServer).Write(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Text_Write_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TextServer).Write(ctx, req.(*WriteRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Text_WriteHTML_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(WriteHTMLRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TextServer).WriteHTML(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Text_WriteHTML_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TextServer).WriteHTML(ctx, req.(*WriteHTMLRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Text_WriteCSS_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(WriteCSSRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TextServer).WriteCSS(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Text_WriteCSS_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TextServer).WriteCSS(ctx, req.(*WriteCSSRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// Text_ServiceDesc is the grpc.ServiceDesc for Text service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var Text_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "multiscope.text.Text",
	HandlerType: (*TextServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "NewWriter",
			Handler:    _Text_NewWriter_Handler,
		},
		{
			MethodName: "NewHTMLWriter",
			Handler:    _Text_NewHTMLWriter_Handler,
		},
		{
			MethodName: "Write",
			Handler:    _Text_Write_Handler,
		},
		{
			MethodName: "WriteHTML",
			Handler:    _Text_WriteHTML_Handler,
		},
		{
			MethodName: "WriteCSS",
			Handler:    _Text_WriteCSS_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "text.proto",
}
