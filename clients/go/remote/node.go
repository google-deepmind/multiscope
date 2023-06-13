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
	pathlib "multiscope/lib/path"
	pb "multiscope/protos/tree_go_proto"
	"sync"

	"github.com/pkg/errors"
)

// Node is a node in the Multiscope tree.
type Node interface {
	// Client returns a client connected to the server owning the node.
	Client() *Client
	// Path returns the path of the node in the tree.
	Path() Path
}

// ClientNode is a structure with a reference to a path and a client.
type ClientNode struct {
	mut         sync.Mutex
	client      *Client
	path        Path
	shouldWrite bool
	id          string
}

// NewClientNode returns a new instance of ClientNode which can be used to implement the Node interface.
func NewClientNode(client *Client, path Path) (*ClientNode, error) {
	node := &ClientNode{client: client, path: path}
	client.Active().Register(path, node.activeCallback)
	var err error
	if node.id, err = pathlib.ToBase64(path.NodePath()); err != nil {
		return nil, errors.Errorf("cannot encode node path to a proto: %v", err)
	}
	return node, nil
}

func (c *ClientNode) activeCallback(path Path, status bool) {
	c.mut.Lock()
	defer c.mut.Unlock()
	c.shouldWrite = status
}

// ShouldWrite if the data written to this is node is expected somewhere.
func (c *ClientNode) ShouldWrite() bool {
	c.mut.Lock()
	defer c.mut.Unlock()
	return c.shouldWrite
}

// Client to which this node is linked to.
func (c *ClientNode) Client() *Client {
	return c.client
}

// Path of the ticker in the tree.
func (c *ClientNode) Path() Path {
	return c.path
}

// NodeID returns an ID for the node that can be shared with the UI client.
func (c *ClientNode) NodeID() string {
	return c.id
}

// Close the node by removing it from the tree.
func (c *ClientNode) Close() error {
	ctx := context.Background()
	_, err := c.Client().TreeClient().Delete(ctx, &pb.DeleteRequest{
		TreeId: c.client.TreeID(),
		Path: &pb.NodePath{
			Path: c.Path(),
		},
	})
	return err
}

// WithShouldWrite is a writer with a ShouldWrite method.
type WithShouldWrite interface {
	ShouldWrite() bool
}
