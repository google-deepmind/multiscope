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
)

type (
	// Marshaler marshals data to store in time lines.
	Marshaler interface {
		// MarshalData writes the node data into a NodeData protocol buffer.
		MarshalData(data *treepb.NodeData, path []string, lastTick uint32)

		// StorageSize returns the size of the storage.
		StorageSize() uint64
	}

	// WithMarshaler provides a custom timeline Marshaler.
	WithMarshaler interface {
		// Timeline returns a marshaler for a timeline.
		Timeline() Marshaler
	}
)

// ToMarshaler converts a node to a marchaler for a timeline.
func ToMarshaler(node core.Node) Marshaler {
	if node == nil {
		return nil
	}
	tl, ok := node.(WithMarshaler)
	if ok {
		return tl.Timeline()
	}
	parent, ok := node.(core.ParentNode)
	if !ok {
		return newConstLeaf(node)
	}
	return NewParent(true, parent)
}
