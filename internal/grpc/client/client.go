// Package client implements helper functions to use the stream gRPC from the client side.
package client

import (
	"context"
	"errors"
	"fmt"

	pb "multiscope/protos/tree_go_proto"
	pbgrpc "multiscope/protos/tree_go_proto"

	"go.uber.org/multierr"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

// PathToNodes sends a request to convert a string path to a list of Nodes.
func PathToNodes(ctx context.Context, clt pbgrpc.TreeClient, paths ...[]string) ([]*pb.Node, error) {
	req := &pb.NodeStructRequest{}
	for _, path := range paths {
		req.Paths = append(req.Paths, &pb.NodePath{Path: path})
	}
	reply, err := clt.GetNodeStruct(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("client.GetNodeStruct returned an error: %w", err)
	}
	if len(paths) != len(reply.Nodes) {
		return nil, fmt.Errorf("unexpected server reply, got %d nodes back, want %d nodes back", len(reply.Nodes), len(paths))
	}
	for i, node := range reply.Nodes {
		if node.Error != "" {
			err = multierr.Append(err, fmt.Errorf("cannot fetch node %d at path %v: %v", i, paths[i], node.Error))
		}
	}
	if err != nil {
		return nil, err
	}
	return reply.Nodes, nil
}

// NodesData sends a request to collect data from a list of nodes.
func NodesData(ctx context.Context, clt pbgrpc.TreeClient, nodes []*pb.Node) ([]*pb.NodeData, error) {
	req := &pb.NodeDataRequest{}
	for _, node := range nodes {
		path := []string{}
		if node.GetPath() != nil {
			path = node.GetPath().GetPath()
		}
		req.Reqs = append(req.Reqs, &pb.DataRequest{
			Path: &pb.NodePath{
				Path: path,
			},
		})
	}
	reply, err := clt.GetNodeData(ctx, req)
	if err != nil {
		return nil, err
	}
	if len(nodes) != len(reply.NodeData) {
		return nil, fmt.Errorf("unexpected server reply, got %d nodes back, want %d nodes back", len(nodes), len(reply.NodeData))
	}
	return reply.NodeData, nil
}

// ToRaw converts data returned by the server into a byte array.
func ToRaw(nodeData *pb.NodeData) ([]byte, error) {
	if nodeData.GetError() != "" {
		return nil, fmt.Errorf("node error: %s", nodeData.GetError())
	}
	raw := nodeData.GetRaw()
	if raw == nil {
		return nil, errors.New("node data does not contain raw data")
	}
	return raw, nil
}

// ToProto converts data returned by the server into a protobuf.
func ToProto(nodeData *pb.NodeData, msg proto.Message) error {
	if nodeData.GetError() != "" {
		return fmt.Errorf("node error: %s", nodeData.GetError())
	}
	p := nodeData.GetPb()
	if p == nil {
		return errors.New("node data does not contain proto data")
	}
	const urlPrefix = "type.googleapis.com/"
	destTypeURL := urlPrefix + string(proto.MessageName(msg))
	if p.TypeUrl != destTypeURL {
		return fmt.Errorf("cannot decode node data because protobuf type url does not match: got %q, want %q", p.TypeUrl, destTypeURL)
	}
	return proto.Unmarshal(p.Value, msg)
}

// SendEvent sends an event to the server given a path.
func SendEvent(ctx context.Context, clt pbgrpc.TreeClient, tickerPath []string, event proto.Message) error {
	pl, err := anypb.New(event)
	if err != nil {
		return err
	}
	_, err = clt.SendEvents(ctx, &pb.SendEventsRequest{
		Events: []*pb.Event{{
			Path: &pb.NodePath{
				Path: tickerPath,
			},
			Payload: pl,
		}},
	})
	return err
}

// Connect a tree client to a running server.
func Connect(addr string) (*grpc.ClientConn, error) {
	conn, err := grpc.Dial(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("client cannot connect to testing server: %w", err)
	}
	return conn, nil
}
