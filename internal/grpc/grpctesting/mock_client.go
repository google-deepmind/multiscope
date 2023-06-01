package grpctesting

import (
	"context"
	pb "multiscope/protos/tree_go_proto"
	pbgrpc "multiscope/protos/tree_go_proto"

	"github.com/pkg/errors"
	"google.golang.org/grpc"
)

// Client a gRPC Tree client for testing.
type Client struct {
	conn   *grpc.ClientConn
	client pbgrpc.TreeClient
	treeID *pb.TreeID
}

func newClient(conn *grpc.ClientConn) (*Client, error) {
	clt := &Client{
		conn:   conn,
		client: pbgrpc.NewTreeClient(conn),
	}
	ctx := context.Background()
	resp, err := clt.client.GetTreeID(ctx, &pb.GetTreeIDRequest{})
	if err != nil {
		return nil, errors.Errorf("cannot fetch a tree ID: %v", err)
	}
	clt.treeID = resp.TreeId
	return clt, nil
}

// TreeClient returns the client used to communicate with the Multiscope server.
func (clt *Client) TreeClient() pbgrpc.TreeClient {
	return clt.client
}

// TreeID returns a the ID of the tree to which this client is connected to.
func (clt *Client) TreeID() *pb.TreeID {
	return clt.treeID
}

// Conn returns the gRPC connection.
func (clt *Client) Conn() *grpc.ClientConn {
	return clt.conn
}
