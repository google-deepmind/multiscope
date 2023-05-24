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

// Package root implements the root of a Multiscope tree.
package root

import (
	"sync"

	"multiscope/internal/server/core"
	"multiscope/internal/server/writers/base"
	rootpb "multiscope/protos/root_go_proto"
	pb "multiscope/protos/tree_go_proto"

	"google.golang.org/protobuf/proto"
)

// Root is the root of the Multiscope tree.
type Root struct {
	mut sync.Mutex
	core.Root
	writer *base.ProtoWriter
	info   *rootpb.RootInfo
}

var _ core.Root = (*Root)(nil)

// NewRoot returns a new Multiscope root node.
func NewRoot() *Root {
	r := &Root{
		Root: base.NewRoot(),
		info: &rootpb.RootInfo{},
	}
	r.writer = base.NewProtoWriter(r.info)
	return r
}

// Path return the path to the root node.
func (r *Root) Path() *core.Path {
	return core.NewPath(r)
}

// MarshalData serializes the data of the root node.
func (r *Root) MarshalData(data *pb.NodeData, path []string, lastTick uint32) {
	if len(path) == 0 {
		r.writer.MarshalData(data, path, lastTick)
	}
	r.Root.MarshalData(data, path, lastTick)
}

func (r *Root) setLayout(layout *rootpb.Layout) error {
	r.mut.Lock()
	defer r.mut.Unlock()
	r.info.Layout = layout
	return r.writer.Write(r.info)
}

func (r *Root) setKeySettings(name string) error {
	r.mut.Lock()
	defer r.mut.Unlock()
	r.info.KeySettings = name
	return r.writer.Write(r.info)
}

func (r *Root) cloneInfo() *rootpb.RootInfo {
	r.mut.Lock()
	defer r.mut.Unlock()
	return proto.Clone(r.info).(*rootpb.RootInfo)
}
