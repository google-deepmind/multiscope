// Package to generate all the files require for Multiscope.
package multiscope

//#go:generate cp $GOROOT/misc/wasm/wasm_exec.js ./web/res

// Generate source code for the protocol buffers and gRPC.

//go:generate protoc -I protos --go_out=.. --go-grpc_out=.. protos/base.proto
//go:generate protoc -I protos --go_out=.. --go-grpc_out=.. protos/root.proto
//go:generate protoc -I protos --go_out=.. --go-grpc_out=.. protos/scalar.proto
//go:generate protoc -I protos --go_out=.. --go-grpc_out=.. protos/table.proto
//go:generate protoc -I protos --go_out=.. --go-grpc_out=.. protos/text.proto
//go:generate protoc -I protos --go_out=.. --go-grpc_out=.. protos/ticker.proto
//go:generate protoc -I protos --go_out=.. --go-grpc_out=.. protos/tree.proto
//go:generate protoc -I protos --go_out=.. --go-grpc_out=.. protos/ui.proto

// Generate web-assembly client.

//go:generate zsh -c "GOOS=js GOARCH=wasm go build -o web/res/multiscope.wasm wasm/mains/multiscope/main.go"
