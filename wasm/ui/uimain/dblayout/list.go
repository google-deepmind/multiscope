package dblayout

import (
	"context"
	"fmt"
	"multiscope/internal/server/core"
	rootpb "multiscope/protos/root_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/ui"

	"honnef.co/go/js/dom/v2"
)

type list struct {
	dbd  ui.Dashboard
	root *dom.HTMLDivElement
	pb   *rootpb.LayoutList
}

func newList(dbd ui.Dashboard) Layout {
	lyt := &list{
		dbd:  dbd,
		pb:   &rootpb.LayoutList{},
		root: dbd.UI().Owner().CreateElement("div").(*dom.HTMLDivElement),
	}
	return lyt
}

func (lyt *list) Load(gLayout *rootpb.Layout) error {
	layout := gLayout.GetList()
	if layout == nil {
		return nil
	}
	nodes, err := lyt.dbd.UI().TreeClient().GetNodeStruct(context.Background(), &treepb.NodeStructRequest{
		Paths: layout.Displayed,
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
	lyt.dbd.UI().Settings().Set(settingKey, &rootpb.Layout{
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
