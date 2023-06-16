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

// Package timeline provides interfaces to serialize nodes in a timeline.
package timeline

import (
	"multiscope/internal/server/core"
	treepb "multiscope/protos/tree_go_proto"

	"google.golang.org/protobuf/proto"
)

type (
	// Marshaler marshals data to store in time lines.
	Marshaler interface {
		// MarshalData writes the node data into a NodeData protocol buffer.
		MarshalData(data *treepb.NodeData, path []string, lastTick uint32)

		// StorageSize returns the size of the storage.
		StorageSize() uint64
	}

	// Timeliner provides a timeline Marshaler from a node in the tree.
	Timeliner interface {
		// Timeline returns a node to store a timeline.
		Timeline() Marshaler
	}

	constProto struct {
		data *treepb.NodeData
	}

	constProtoWithChildren struct {
		Marshaler
		core.Parent
	}
)

var (
	_ Marshaler   = (*constProto)(nil)
	_ Marshaler   = (*constProtoWithChildren)(nil)
	_ core.Parent = (*constProtoWithChildren)(nil)
)

func (m *constProto) MarshalData(data *treepb.NodeData, path []string, lastTick uint32) {
	proto.Merge(data, m.data)
}

func (m *constProto) StorageSize() uint64 {
	return uint64(proto.Size(m.data))
}

func (m *constProtoWithChildren) MarshalData(data *treepb.NodeData, path []string, lastTick uint32) {
	m.Marshaler.MarshalData(data, path, lastTick)
}

func (m *constProtoWithChildren) StorageSize() uint64 {
	return m.Marshaler.StorageSize()
}

// Child returns a child node given its ID.
func (m *constProtoWithChildren) Child(name string) (core.Node, error) {
	return m.Parent.Child(name)
}

// Children returns the (sorted) list of names of all children of this node.
func (m *constProtoWithChildren) Children() ([]string, error) {
	return m.Parent.Children()
}

// ToMarshaler converts a node to a marchaler for a timeline.
func ToMarshaler(path *treepb.NodePath, node core.Node) Marshaler {
	if node == nil {
		return nil
	}
	tl, ok := node.(Timeliner)
	if ok {
		return tl.Timeline()
	}

	cstPB := &constProto{data: &treepb.NodeData{Path: path}}
	node.MarshalData(cstPB.data, nil, 0)
	parent, ok := node.(core.Parent)
	if !ok {
		return cstPB
	}
	return &constProtoWithChildren{
		Marshaler: cstPB,
		Parent:    parent,
	}
}

// ToErrorMarshaler stores an error in a marshaler.
func ToErrorMarshaler(path *treepb.NodePath, err error) Marshaler {
	return &constProto{
		data: &treepb.NodeData{
			Path:  path,
			Error: err.Error(),
		},
	}
}
