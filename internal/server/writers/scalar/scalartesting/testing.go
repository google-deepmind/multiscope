// Package scalartesting provides functions to test scalar writers.
package scalartesting

import (
	"context"
	"fmt"

	"multiscope/internal/grpc/client"
	"multiscope/internal/mime"
	"multiscope/internal/server/writers/scalar"
	plotpb "multiscope/protos/plot_go_proto"
	pbgrpc "multiscope/protos/tree_go_proto"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protocmp"
)

func init() {
	scalar.SetHistoryLength(2)
}

// Scalar01Name is the name of the scalar writer in the tree.
const Scalar01Name = "scalar01"

// Scalar01Data is the data to write to the scalar01 writer.
var Scalar01Data = []map[string]float64{
	{"a": 1, "b": 2},
	{"a": 3, "b": 4},
	{"a": 5, "b": 6},
}

func buildWant() *plotpb.ScalarsPlot {
	return &plotpb.ScalarsPlot{Plot: &plotpb.Plot{Plotters: []*plotpb.Plotter{
		{
			Legend: "a",
			Serie: &plotpb.Serie{Points: []*plotpb.Point{
				{X: 1, Y: 3},
				{X: 2, Y: 5},
			}},
		},
		{
			Legend: "b",
			Serie: &plotpb.Serie{Points: []*plotpb.Point{
				{X: 1, Y: 4},
				{X: 2, Y: 6},
			}},
		},
	}}}
}

// CheckScalar01 checks the data that can be read from the scalar01 node.
func CheckScalar01(clt pbgrpc.TreeClient, path []string) error {
	ctx := context.Background()
	// Get the nodes and check their types.
	nodes, err := client.PathToNodes(ctx, clt, path)
	if err != nil {
		return err
	}
	tsGot := &plotpb.ScalarsPlot{}
	if diff := cmp.Diff(nodes[0].GetMime(), mime.NamedProtobuf(string(proto.MessageName(tsGot)))); len(diff) > 0 {
		return fmt.Errorf("mime type error: %s", diff)
	}

	// Check the data returned by the server.
	data, err := client.NodesData(ctx, clt, nodes)
	if err != nil {
		return err
	}
	err = client.ToProto(data[0], tsGot)
	if err != nil {
		return err
	}
	tsWant := buildWant()
	if diff := cmp.Diff(tsGot, tsWant, protocmp.Transform()); len(diff) > 0 {
		return fmt.Errorf("time series do not match:\n%s\ngot: %T\n%s\nwant: %T\n%s", diff, tsGot, prototext.Format(tsGot), tsWant, prototext.Format(tsWant))
	}
	return nil
}
