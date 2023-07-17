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

// Package client implements helper functions to use the stream gRPC from the client side.
package client

import (
	"context"
	"fmt"

	pb "multiscope/protos/tree_go_proto"
	pbgrpc "multiscope/protos/tree_go_proto"

	"github.com/pkg/errors"
	"go.uber.org/multierr"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

// Client is a client to a tree service.
type Client interface {
	// TreeID returns the ID of the tree.
	TreeID() *pb.TreeID
	// TreeClient returns the gRPC client.
	TreeClient() pbgrpc.TreeClient
}

// PathToNodes sends a request to convert a string path to a list of Nodes.
func PathToNodes(ctx context.Context, clt Client, paths ...[]string) (_ []*pb.Node, err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("cannot fetch nodes from path: %w", err)
		}
	}()
	req := &pb.NodeStructRequest{TreeId: clt.TreeID()}
	for _, path := range paths {
		req.Paths = append(req.Paths, &pb.NodePath{Path: path})
	}
	reply, err := clt.TreeClient().GetNodeStruct(ctx, req)
	if err != nil {
		return nil, errors.Errorf("client.GetNodeStruct returned an error: %v", err)
	}
	if len(paths) != len(reply.Nodes) {
		return nil, errors.Errorf("unexpected server reply, got %d nodes back, want %d nodes back", len(reply.Nodes), len(paths))
	}
	for i, node := range reply.Nodes {
		if node.Error != "" {
			err = multierr.Append(err, fmt.Errorf("cannot fetch node %d at path %v: %v", i, paths[i], node.Error))
		}
	}
	if err != nil {
		return nil, errors.Errorf("%v", err)
	}
	return reply.Nodes, nil
}

// NodesData sends a request to collect data from a list of nodes.
func NodesData(ctx context.Context, clt Client, nodes []*pb.Node) ([]*pb.NodeData, error) {
	req := &pb.NodeDataRequest{TreeId: clt.TreeID()}
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
	reply, err := clt.TreeClient().GetNodeData(ctx, req)
	if err != nil {
		return nil, errors.Errorf("cannot fetch data from nodes: %v", err)
	}
	if len(nodes) != len(reply.NodeData) {
		return nil, errors.Errorf("unexpected server reply, got %d nodes back, want %d nodes back", len(nodes), len(reply.NodeData))
	}
	return reply.NodeData, nil
}

// ToRaw converts data returned by the server into a byte array.
func ToRaw(nodeData *pb.NodeData) ([]byte, error) {
	if nodeData.GetError() != "" {
		return nil, errors.Errorf("node error: %s", nodeData.GetError())
	}
	raw := nodeData.GetRaw()
	if raw == nil {
		return nil, errors.Errorf("node data does not contain raw data")
	}
	return raw, nil
}

// ToProto converts data returned by the server into a protobuf.
func ToProto(nodeData *pb.NodeData, msg proto.Message) error {
	if nodeData.GetError() != "" {
		return errors.Errorf("node error: %s", nodeData.GetError())
	}
	p := nodeData.GetPb()
	if p == nil {
		return errors.Errorf("node data does not contain proto data")
	}
	const urlPrefix = "type.googleapis.com/"
	destTypeURL := urlPrefix + string(proto.MessageName(msg))
	if p.TypeUrl != destTypeURL {
		return errors.Errorf("cannot decode node data because protobuf type url does not match: got %q, want %q", p.TypeUrl, destTypeURL)
	}
	return proto.Unmarshal(p.Value, msg)
}

// SendEvent sends an event to the server given a path.
func SendEvent(ctx context.Context, clt Client, tickerPath []string, event proto.Message) error {
	pl, err := anypb.New(event)
	if err != nil {
		return err
	}
	_, err = clt.TreeClient().SendEvents(ctx, &pb.SendEventsRequest{
		TreeId: clt.TreeID(),
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
		return nil, errors.Errorf("client cannot connect to testing server: %v", err)
	}
	return conn, nil
}
