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

package base

import (
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"sync"

	"multiscope/internal/server/core"
	"multiscope/internal/server/treeservice"
	pb "multiscope/protos/tree_go_proto"

	"github.com/pkg/errors"
)

// Group implements a Parent node to group a set of node together.
type Group struct {
	mux      sync.Mutex
	children map[string]core.Node
	mime     string
}

var _ core.Parent = (*Group)(nil)

// NewGroup returns a new Group parent node.
func NewGroup(mime string) *Group {
	return &Group{
		children: make(map[string]core.Node),
		mime:     mime,
	}
}

// AddToTree adds the proto writer to tree in server state, at the given path.
func (g *Group) AddToTree(state *treeservice.State, path *pb.NodePath) (*core.Path, error) {
	return core.SetNodeAt(state.Root(), path, g)
}

// Children returns the list of children of this node.
func (g *Group) Children() ([]string, error) {
	g.mux.Lock()
	defer g.mux.Unlock()
	names := []string{}
	for name := range g.children {
		names = append(names, name)
	}
	sort.Strings(names)
	return names, nil
}

// AddChild adds a child to the receiver group.
func (g *Group) AddChild(name string, child core.Node) string {
	g.mux.Lock()
	defer g.mux.Unlock()
	// If the name is taken, rename by appending a number, eg child1, child2, etc.
	i := 1
	nodeName := name
	for _, ok := g.children[nodeName]; ok; i++ {
		nodeName = name + strconv.Itoa(i)
		_, ok = g.children[nodeName]
	}
	g.children[nodeName] = child
	return nodeName
}

// DeleteChild deletes a child in the receiver group.
// If the child implements io.Close, calls the method before removing the node from the group.
func (g *Group) DeleteChild(childName string) error {
	g.mux.Lock()
	defer g.mux.Unlock()
	node := g.children[childName]
	if node == nil {
		return errors.Errorf("%q is not a child of this group (available children are: %v)", childName, core.ChildrenNames(g))
	}
	closer, ok := node.(io.Closer)
	if ok {
		if err := closer.Close(); err != nil {
			return err
		}
	}
	delete(g.children, childName)
	return nil
}

// Child returns a child node given its ID.
// Returns a nil node if the name does not match any existing children name.
// Returns an error if getting the list of children triggered an error.
func (g *Group) Child(name string) (core.Node, error) {
	g.mux.Lock()
	defer g.mux.Unlock()
	return g.children[name], nil
}

// MarshalData returns data of the child denoted by the path recursively.
func (g *Group) MarshalData(data *pb.NodeData, path []string, lastTick uint32) {
	if len(path) == 0 {
		return
	}
	childName := path[0]
	child, err := g.Child(childName)
	if err != nil {
		data.Error = err.Error()
		return
	}
	if child == nil {
		data.Error = fmt.Sprintf("%q in path '%s' is not a child of this group (available children are: %v)", childName, path, core.ChildrenNames(g))
		return
	}
	child.MarshalData(data, path[1:], lastTick)
}

func indent(r string) string {
	if r == "" {
		return r
	}
	rs := strings.Split(r, "\n")
	filtered := []string{}
	for _, s := range rs {
		if s == "" {
			continue
		}
		filtered = append(filtered, s)
	}
	const prefix = "  "
	return prefix + strings.Join(filtered, "\n"+prefix)
}

// MIME returns the MIME type of this node.
func (g *Group) MIME() string {
	return g.mime
}

func (g *Group) String() string {
	r := ""
	for name, child := range g.children {
		childS := fmt.Sprintf("|-%s [%T]: %v", name, child, child)
		r += childS + "\n"
	}
	return g.mime + "\n" + indent(r) + "\n"
}
