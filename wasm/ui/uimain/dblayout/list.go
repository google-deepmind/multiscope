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

package dblayout

import (
	"context"
	"fmt"
	"multiscope/internal/server/core"
	rootpb "multiscope/protos/root_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	uipb "multiscope/protos/ui_go_proto"
	"multiscope/wasm/ui"

	"honnef.co/go/js/dom/v2"
)

const defaultRowHeight = 300

type list struct {
	dbd  ui.Dashboard
	root *dom.HTMLDivElement
	pb   *rootpb.LayoutList
}

func newList(dbd ui.Dashboard) Layout {
	lyt := &list{
		dbd:  dbd,
		root: dbd.UI().Owner().Doc().CreateElement("div").(*dom.HTMLDivElement),
	}
	lyt.root.Class().Add("panels_layout_list")
	lyt.fixProto()
	return lyt
}

func (lyt *list) fixProto() {
	if lyt.pb == nil {
		lyt.pb = &rootpb.LayoutList{}
	}
	if lyt.pb.DefaultRowHeight == 0 {
		lyt.pb.DefaultRowHeight = defaultRowHeight
	}
}

func (lyt *list) PreferredSize() *uipb.ElementSize {
	return &uipb.ElementSize{
		Height: lyt.pb.DefaultRowHeight,
		Width:  0,
	}
}

func (lyt *list) Load(gLayout *rootpb.Layout) error {
	layout := gLayout.GetList()
	if layout == nil {
		return nil
	}
	lyt.pb.DefaultRowHeight = layout.DefaultRowHeight
	lyt.fixProto()

	treeClient, treeID := lyt.dbd.UI().TreeClient()
	nodes, err := treeClient.GetNodeStruct(context.Background(), &treepb.NodeStructRequest{
		TreeId: treeID,
		Paths:  layout.Displayed,
	})
	if err != nil {
		return fmt.Errorf("cannot query the tree structure: %v", err)
	}
	for _, node := range nodes.Nodes {
		if err := lyt.dbd.OpenPanel(node); err != nil {
			return err
		}
	}
	return nil
}

func (lyt *list) Root() dom.Node {
	return lyt.root
}

func (lyt *list) store() {
	lyt.dbd.UI().Settings().Set(lyt, SettingKey, &rootpb.Layout{
		Layout: &rootpb.Layout_List{
			List: lyt.pb,
		},
	})
}

func (lyt *list) Append(pnl ui.Panel) {
	lyt.Root().AppendChild(pnl.Root())
	path := pnl.Desc().Path()
	if path != nil {
		lyt.pb.Displayed = append(lyt.pb.Displayed, pnl.Desc().Path())
	}
	lyt.store()
}

func findPath(paths []*treepb.NodePath, path *treepb.NodePath) int {
	k := core.ToKey(path.Path)
	for i, p := range paths {
		if core.ToKey(p.Path) == k {
			return i
		}
	}
	return -1
}

func (lyt *list) Remove(pnl ui.Panel) {
	lyt.Root().RemoveChild(pnl.Root())
	path := pnl.Desc().Path()
	if path != nil {
		i := findPath(lyt.pb.Displayed, path)
		lyt.pb.Displayed = append(lyt.pb.Displayed[:i], lyt.pb.Displayed[i+1:]...)
	}
	lyt.store()
}
