// Protocol buffer to stream scientific data.

// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.3.0
// - protoc             v3.21.12
// source: ticker.proto

package ticker_go_proto

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
	Tickers_NewTicker_FullMethodName   = "/multiscope.ticker.Tickers/NewTicker"
	Tickers_WriteTicker_FullMethodName = "/multiscope.ticker.Tickers/WriteTicker"
	Tickers_NewPlayer_FullMethodName   = "/multiscope.ticker.Tickers/NewPlayer"
	Tickers_StoreFrame_FullMethodName  = "/multiscope.ticker.Tickers/StoreFrame"
)

// TickersClient is the client API for Tickers service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type TickersClient interface {
	// Create a new ticker node in Multiscope.
	NewTicker(ctx context.Context, in *NewTickerRequest, opts ...grpc.CallOption) (*NewTickerResponse, error)
	// Write ticker data.
	WriteTicker(ctx context.Context, in *WriteTickerRequest, opts ...grpc.CallOption) (*WriteTickerResponse, error)
	// Create a new player node in Multiscope.
	NewPlayer(ctx context.Context, in *NewPlayerRequest, opts ...grpc.CallOption) (*NewPlayerResponse, error)
	// Write ticker data.
	StoreFrame(ctx context.Context, in *StoreFrameRequest, opts ...grpc.CallOption) (*StoreFrameResponse, error)
}

type tickersClient struct {
	cc grpc.ClientConnInterface
}

func NewTickersClient(cc grpc.ClientConnInterface) TickersClient {
	return &tickersClient{cc}
}

func (c *tickersClient) NewTicker(ctx context.Context, in *NewTickerRequest, opts ...grpc.CallOption) (*NewTickerResponse, error) {
	out := new(NewTickerResponse)
	err := c.cc.Invoke(ctx, Tickers_NewTicker_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *tickersClient) WriteTicker(ctx context.Context, in *WriteTickerRequest, opts ...grpc.CallOption) (*WriteTickerResponse, error) {
	out := new(WriteTickerResponse)
	err := c.cc.Invoke(ctx, Tickers_WriteTicker_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *tickersClient) NewPlayer(ctx context.Context, in *NewPlayerRequest, opts ...grpc.CallOption) (*NewPlayerResponse, error) {
	out := new(NewPlayerResponse)
	err := c.cc.Invoke(ctx, Tickers_NewPlayer_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *tickersClient) StoreFrame(ctx context.Context, in *StoreFrameRequest, opts ...grpc.CallOption) (*StoreFrameResponse, error) {
	out := new(StoreFrameResponse)
	err := c.cc.Invoke(ctx, Tickers_StoreFrame_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// TickersServer is the server API for Tickers service.
// All implementations must embed UnimplementedTickersServer
// for forward compatibility
type TickersServer interface {
	// Create a new ticker node in Multiscope.
	NewTicker(context.Context, *NewTickerRequest) (*NewTickerResponse, error)
	// Write ticker data.
	WriteTicker(context.Context, *WriteTickerRequest) (*WriteTickerResponse, error)
	// Create a new player node in Multiscope.
	NewPlayer(context.Context, *NewPlayerRequest) (*NewPlayerResponse, error)
	// Write ticker data.
	StoreFrame(context.Context, *StoreFrameRequest) (*StoreFrameResponse, error)
	mustEmbedUnimplementedTickersServer()
}

// UnimplementedTickersServer must be embedded to have forward compatible implementations.
type UnimplementedTickersServer struct {
}

func (UnimplementedTickersServer) NewTicker(context.Context, *NewTickerRequest) (*NewTickerResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method NewTicker not implemented")
}
func (UnimplementedTickersServer) WriteTicker(context.Context, *WriteTickerRequest) (*WriteTickerResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method WriteTicker not implemented")
}
func (UnimplementedTickersServer) NewPlayer(context.Context, *NewPlayerRequest) (*NewPlayerResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method NewPlayer not implemented")
}
func (UnimplementedTickersServer) StoreFrame(context.Context, *StoreFrameRequest) (*StoreFrameResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method StoreFrame not implemented")
}
func (UnimplementedTickersServer) mustEmbedUnimplementedTickersServer() {}

// UnsafeTickersServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to TickersServer will
// result in compilation errors.
type UnsafeTickersServer interface {
	mustEmbedUnimplementedTickersServer()
}

func RegisterTickersServer(s grpc.ServiceRegistrar, srv TickersServer) {
	s.RegisterService(&Tickers_ServiceDesc, srv)
}

func _Tickers_NewTicker_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NewTickerRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TickersServer).NewTicker(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Tickers_NewTicker_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TickersServer).NewTicker(ctx, req.(*NewTickerRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Tickers_WriteTicker_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(WriteTickerRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TickersServer).WriteTicker(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Tickers_WriteTicker_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TickersServer).WriteTicker(ctx, req.(*WriteTickerRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Tickers_NewPlayer_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(NewPlayerRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TickersServer).NewPlayer(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Tickers_NewPlayer_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TickersServer).NewPlayer(ctx, req.(*NewPlayerRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Tickers_StoreFrame_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(StoreFrameRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(TickersServer).StoreFrame(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Tickers_StoreFrame_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(TickersServer).StoreFrame(ctx, req.(*StoreFrameRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// Tickers_ServiceDesc is the grpc.ServiceDesc for Tickers service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var Tickers_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "multiscope.ticker.Tickers",
	HandlerType: (*TickersServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "NewTicker",
			Handler:    _Tickers_NewTicker_Handler,
		},
		{
			MethodName: "WriteTicker",
			Handler:    _Tickers_WriteTicker_Handler,
		},
		{
			MethodName: "NewPlayer",
			Handler:    _Tickers_NewPlayer_Handler,
		},
		{
			MethodName: "StoreFrame",
			Handler:    _Tickers_StoreFrame_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "ticker.proto",
}
