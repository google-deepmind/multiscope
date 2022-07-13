package remote

import (
	"context"
	pb "multiscope/protos/base_go_proto"
	pbgrpc "multiscope/protos/base_go_proto"

	"github.com/pkg/errors"
)

// Group is a folder in the tree.
type Group struct {
	*ClientNode
	clt pbgrpc.BaseWritersClient
}

// NewGroup creates a new group in the tree.
func NewGroup(clt *Client, name string, parent Path) (*Group, error) {
	clw := pbgrpc.NewBaseWritersClient(clt.Connection())
	ctx := context.Background()
	path := clt.toChildPath(name, parent)
	rep, err := clw.NewGroup(ctx, &pb.NewGroupRequest{
		Path: path.NodePath(),
	})
	if err != nil {
		return nil, err
	}
	grp := rep.GetGrp()
	if grp == nil {
		return nil, errors.New("server has returned a nil group")
	}
	return &Group{
		ClientNode: NewClientNode(clt, toPath(grp)),
		clt:        clw,
	}, nil
}
