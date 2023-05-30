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

package ticker

import (
	"multiscope/internal/mime"
	"multiscope/internal/server/core"
	"multiscope/internal/server/treeservice"
	"multiscope/internal/server/writers/base"
	tickerpb "multiscope/protos/ticker_go_proto"
	treepb "multiscope/protos/tree_go_proto"
)

// Ticker is a group node that displays timing data.
type Ticker struct {
	*base.Group
	writer *base.ProtoWriter
}

var (
	_ core.Node       = (*Ticker)(nil)
	_ core.Parent     = (*Ticker)(nil)
	_ core.ChildAdder = (*Ticker)(nil)
)

// NewTicker returns a new writer to stream data tables.
func NewTicker() *Ticker {
	return &Ticker{
		Group:  base.NewGroup(mime.MultiscopeTicker),
		writer: base.NewProtoWriter(&tickerpb.TickerData{}),
	}
}

func (t *Ticker) addToTree(state *treeservice.State, path *treepb.NodePath) (*core.Path, error) {
	return core.SetNodeAt(state.Root(), path, t)
}

// Write the latest tick data.
func (t *Ticker) Write(data *tickerpb.TickerData) error {
	return t.writer.Write(data)
}

// MIME returns the mime type of this node.
func (t *Ticker) MIME() string {
	return t.writer.MIME()
}

// MarshalData writes the node data into a NodeData protocol buffer.
func (t *Ticker) MarshalData(data *treepb.NodeData, path []string, lastTick uint32) {
	if len(path) == 0 {
		t.writer.MarshalData(data, path, lastTick)
		return
	}
	t.Group.MarshalData(data, path, lastTick)
}
