// Package basetesting contains utility methods for service_test.go
package basetesting

import (
	"bytes"
	"context"
	"fmt"

	"multiscope/internal/grpc/client"
	pbgrpc "multiscope/protos/tree_go_proto"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// Proto01Name is the name of the proto writer in the tree.
const Proto01Name = "proto01"

// Proto01Data is the data to write to the writer.
var Proto01Data = []*timestamppb.Timestamp{
	{
		Seconds: 1,
	},
	{
		Seconds: 2,
	},
	{
		Seconds: 3,
	},
}

// TimestampToAny convert a timestamp proto into an Any proto.
func TimestampToAny(ts *timestamppb.Timestamp) (an *anypb.Any, err error) {
	an = &anypb.Any{}
	err = an.MarshalFrom(ts)
	return
}

// CheckProto01 checks the data exported by a proto node.
func CheckProto01(ctx context.Context, clt pbgrpc.TreeClient, path []string, i int) error {
	nodes, err := client.PathToNodes(ctx, clt, path)
	if err != nil {
		return err
	}

	data, err := client.NodesData(ctx, clt, nodes)
	if err != nil {
		return err
	}
	pb := &timestamppb.Timestamp{}
	err = client.ToProto(data[0], pb)
	if err != nil {
		return err
	}
	want := Proto01Data[i]
	if !proto.Equal(pb, want) {
		return fmt.Errorf("proto error in test %d: got %s, want %s", i, pb, want)
	}
	return nil
}

// Raw01Name is the name of the raw writer in the tree.
const Raw01Name = "raw01"

// Raw01Data is the data to write to the writer.
var Raw01Data = [][]byte{
	[]byte("test0"),
	[]byte("test1"),
	[]byte("test2"),
}

// CheckRaw01 checks the data exported by a raw node.
func CheckRaw01(ctx context.Context, clt pbgrpc.TreeClient, path []string, i int) error {
	nodes, err := client.PathToNodes(ctx, clt, path)
	if err != nil {
		return err
	}

	data, err := client.NodesData(ctx, clt, nodes)
	if err != nil {
		return err
	}
	raw, err := client.ToRaw(data[0])
	if err != nil {
		return err
	}
	want := Raw01Data[i]
	if !bytes.Equal(raw, want) {
		return fmt.Errorf("raw error in test %d: got %s, want %s", i, raw, want)
	}
	return nil
}
