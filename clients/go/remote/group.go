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
	path := clt.toChildPath(name, parent)
	rep, err := clw.NewGroup(context.Background(), &pb.NewGroupRequest{
		TreeId: clt.TreeID(),
		Path:   path.NodePath(),
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

// Root returns a root node.
func Root(clt *Client) *Group {
	return &Group{ClientNode: NewClientNode(clt, nil)}
}
