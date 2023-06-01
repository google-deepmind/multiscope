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

package timedb

import (
	"multiscope/internal/server/core"
	"multiscope/internal/server/core/timeline"
	treepb "multiscope/protos/tree_go_proto"
)

type context struct {
	current core.Node
	path    *core.Path
}

// NewConstructor returns a new constructor to construct a new record.
func (db *TimeDB) newContext() *context {
	return &context{current: db.node, path: core.NewPath(db.node)}
}

func (ctx *context) buildChildRecord(parent core.Parent, childName string) *Record {
	child, childErr := parent.Child(childName)
	if childErr != nil {
		return newErrorRecord(childErr)
	}
	childCtx := &context{
		current: child,
		path:    ctx.path.PathTo(childName),
	}
	return childCtx.buildRecord()
}

func (ctx *context) buildRecord() *Record {
	if current, ok := ctx.current.(timeline.Adapter); ok {
		ctx.current = current.Timeline()
	}

	// Serialize the data for the current node.
	data := &treepb.NodeData{Path: ctx.path.PB()}
	ctx.current.MarshalData(data, nil, 0)
	rec := newRecord(data)

	// Check if the node has children.
	parent, ok := ctx.current.(core.Parent)
	if !ok {
		return rec
	}

	// Process children.
	children, err := parent.Children()
	if err != nil {
		rec.setFetchChildrenError(err)
		return rec
	}
	for _, childName := range children {
		childRec := ctx.buildChildRecord(parent, childName)
		rec.addChild(childName, childRec)
	}
	return rec
}
