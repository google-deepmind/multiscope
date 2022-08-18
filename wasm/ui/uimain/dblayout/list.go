package dblayout

import (
	"context"
	"fmt"
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

func (lyt *list) Append(pnl ui.Panel) {
	lyt.Root().AppendChild(pnl.Root())
}

func (lyt *list) Remove(pnl ui.Panel) {
	lyt.Root().RemoveChild(pnl.Root())
}
