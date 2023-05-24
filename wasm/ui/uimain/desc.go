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

package uimain

import (
	treepb "multiscope/protos/tree_go_proto"
	uipb "multiscope/protos/ui_go_proto"
	"multiscope/wasm/renderers"
	"multiscope/wasm/ui"
	"syscall/js"
)

// Descriptor stores how to get and process data for a panel.
type Descriptor struct {
	dbd           *Dashboard
	node          *treepb.Node // can be nil
	pb            uipb.Panel
	transferables map[string]any
}

func rootDescriptor() *Descriptor {
	return &Descriptor{
		pb: uipb.Panel{
			Id:    0,
			Paths: []*treepb.NodePath{{}},
		}}
}

var nextID = uint32(1)

// NewDescriptor returns a new info descriptor to assign to a panel.
func (dbd *Dashboard) NewDescriptor(node *treepb.Node, renderer renderers.Newer, paths ...*treepb.NodePath) ui.Descriptor {
	id := nextID
	nextID++
	return &Descriptor{
		node: node,
		dbd:  dbd,
		pb: uipb.Panel{
			Id:       id,
			Paths:    paths,
			Renderer: renderers.Name(renderer),
		}}
}

// AddTransferable adds a key,value pair to transfer to the renderer worker.
func (dsc *Descriptor) AddTransferable(name string, v js.Value) {
	if dsc.transferables == nil {
		dsc.transferables = make(map[string]any)
	}
	dsc.transferables[name] = v
}

// ID returns the ID of the panel.
func (dsc *Descriptor) ID() ui.PanelID {
	return ui.PanelID(dsc.pb.Id)
}

// Path returns the path of the node in the tree.
// Can return nil.
func (dsc *Descriptor) Path() *treepb.NodePath {
	if dsc.node == nil {
		return nil
	}
	return dsc.node.Path
}

// PanelPB returns the list of path necessary for the panel.
func (dsc *Descriptor) PanelPB() (*uipb.Panel, map[string]any) {
	return &dsc.pb, dsc.transferables
}

// Dashboard returns the owner of the descriptor.
func (dsc *Descriptor) Dashboard() ui.Dashboard {
	return dsc.dbd
}
