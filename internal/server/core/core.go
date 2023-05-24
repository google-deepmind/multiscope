// Copyright 2023 Google LLC
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

// Package core implements a graph of scientific data streams.
package core

import (
	"errors"
	"fmt"
	pb "multiscope/protos/tree_go_proto"
	"reflect"
)

type (
	// Node represents a node (parent or leaf) in the graph.
	Node interface {
		MIME() string

		// MarshalData writes the node data into a NodeData protocol buffer.
		MarshalData(data *pb.NodeData, path []string, lastTick uint32)
	}

	// Parent is node in the graph with children.
	Parent interface {
		Node
		// Child returns a child node given its ID.
		Child(name string) (Node, error)
		// Children returns the (sorted) list of names of all children of this node.
		Children() ([]string, error)
	}

	// ChildAdder adds children nodes to a parent.
	ChildAdder interface {
		// AddChild adds a node to a parent.
		// The parent can decide to use a different name if a child already exists with the same name.
		// The function returns the name used by the parent.
		AddChild(name string, node Node) string
	}

	// Root is the root of a stream tree node.
	Root interface {
		Parent

		ChildAdder

		Path() *Path

		// Delete a node in the tree.
		Delete(path []string) error
	}
)

// WithPBPath returns a path in the tree as a stream Path protocol buffer.
// Typically, Multiscope GRPC writers will implement that interface.
type WithPBPath interface {
	GetPath() *pb.NodePath
}

// Set finds a path starting from parent and set dst to the node found in the tree.
// The code returns an error if the node in the tree cannot be cast to dst.
func Set(dst any, parent Parent, withPath WithPBPath) error {
	if withPath == nil {
		return errors.New("cannot get a path in the tree from nil")
	}
	path := withPath.GetPath()
	if path == nil {
		return fmt.Errorf("%T has its path set to nil", withPath)
	}
	node, err := PathToNode(parent, path.GetPath())
	if err != nil {
		return err
	}
	srcType := reflect.TypeOf(node)
	dstVal := reflect.ValueOf(dst).Elem()
	dstType := dstVal.Type()
	if !srcType.ConvertibleTo(dstType) {
		return fmt.Errorf("cannot cast node at path %v of type %s to type %s", path, srcType.String(), dstType.String())
	}
	dstVal.Set(reflect.ValueOf(node).Convert(srcType))
	return nil
}
