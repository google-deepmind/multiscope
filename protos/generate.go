// Package protos defines all the protocol buffers and gRPC services used by Multiscope.
package protos

// Generate source code for the protocol buffers and gRPC.

//go:generate protoc -I ../protos --go_out=../.. --go-grpc_out=../.. ../protos/base.proto
//go:generate protoc -I ../protos --go_out=../.. --go-grpc_out=../.. ../protos/root.proto
//go:generate protoc -I ../protos --go_out=../.. --go-grpc_out=../.. ../protos/scalar.proto
//go:generate protoc -I ../protos --go_out=../.. --go-grpc_out=../.. ../protos/table.proto
//go:generate protoc -I ../protos --go_out=../.. --go-grpc_out=../.. ../protos/text.proto
//go:generate protoc -I ../protos --go_out=../.. --go-grpc_out=../.. ../protos/ticker.proto
//go:generate protoc -I ../protos --go_out=../.. --go-grpc_out=../.. ../protos/tree.proto
//go:generate protoc -I ../protos --go_out=../.. --go-grpc_out=../.. ../protos/ui.proto
