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

// Package protos defines all the protocol buffers and gRPC services used by Multiscope.
package protos

// Generate source code for the protocol buffers and gRPC.

//go:generate protoc -I ../protos --go_out=../.. --go-grpc_out=../.. ../protos/base.proto
//go:generate protoc -I ../protos --go_out=../.. --go-grpc_out=../.. ../protos/events.proto
//go:generate protoc -I ../protos --go_out=../.. --go-grpc_out=../.. ../protos/plot.proto
//go:generate protoc -I ../protos --go_out=../.. --go-grpc_out=../.. ../protos/root.proto
//go:generate protoc -I ../protos --go_out=../.. --go-grpc_out=../.. ../protos/scalar.proto
//go:generate protoc -I ../protos --go_out=../.. --go-grpc_out=../.. ../protos/tensor.proto
//go:generate protoc -I ../protos --go_out=../.. --go-grpc_out=../.. ../protos/text.proto
//go:generate protoc -I ../protos --go_out=../.. --go-grpc_out=../.. ../protos/ticker.proto
//go:generate protoc -I ../protos --go_out=../.. --go-grpc_out=../.. ../protos/tree.proto
//go:generate protoc -I ../protos --go_out=../.. --go-grpc_out=../.. ../protos/ui.proto

// Generate the API version.
//go:generate zsh ./generate_version.zsh
