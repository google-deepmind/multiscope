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
	if lyt.pb.DefaultPanelSize == nil {
		lyt.pb.DefaultPanelSize = &rootpb.LayoutList_Size{}
	}
	if lyt.pb.DefaultPanelSize.Width == 0 {
		lyt.pb.DefaultPanelSize.Width = defaultPanelWidth
	}
	if lyt.pb.DefaultPanelSize.Height == 0 {
		lyt.pb.DefaultPanelSize.Height = defaultPanelHeight
	}
}

func (lyt *list) Load(gLayout *rootpb.Layout) error {
	layout := gLayout.GetList()
	if layout == nil {
		return nil
	}
	lyt.pb.DefaultPanelSize = layout.DefaultPanelSize
	lyt.fixProto()

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
	lyt.dbd.UI().Settings().Set(lyt, SettingKey, &rootpb.Layout{
		Layout: &rootpb.Layout_List{
			List: lyt.pb,
		},
	})
}

func (lyt *list) format(div *dom.HTMLDivElement) {
	height := fmt.Sprintf("%dpx", lyt.pb.DefaultPanelSize.Height)
	div.Style().SetProperty("height", height, "")
	width := fmt.Sprintf("%dpx", lyt.pb.DefaultPanelSize.Width)
	div.Style().SetProperty("width", width, "")
}

func (lyt *list) Append(pnl ui.Panel) {
	lyt.format(pnl.Root())
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
