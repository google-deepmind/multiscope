package uimain

import (
	"context"
	"fmt"
	"multiscope/internal/mime"
	rootpb "multiscope/protos/root_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	uipb "multiscope/protos/ui_go_proto"
	"multiscope/wasm/ui"

	"github.com/pkg/errors"
	"honnef.co/go/js/dom/v2"
)

// Dashboard gets the data sent by the renderers (from a webworker) and display
// them in the main UI thread.
type Dashboard struct {
	ui     *UI
	panels map[ui.PanelID]ui.Panel
	root   dom.Node
}

func newDashboard(main *UI) (*Dashboard, error) {
	const dashboardClass = "container__middle"
	elements := main.Owner().GetElementsByClassName(dashboardClass)
	if len(elements) != 1 {
		return nil, errors.Errorf("wrong number of elements of class %q: got %d but want 1", dashboardClass, len(elements))
	}
	return &Dashboard{
		ui:     main,
		panels: make(map[ui.PanelID]ui.Panel),
		root:   elements[0],
	}, nil
}

func (dbd *Dashboard) descriptor() *Descriptor {
	return rootDescriptor()
}

// UI returns the UI instance owning the dashboard.
func (dbd *Dashboard) UI() ui.UI {
	return dbd.ui
}

func (dbd *Dashboard) updateLayout(layout *rootpb.Layout) error {
	nodes, err := dbd.ui.treeClient.GetNodeStruct(context.Background(), &treepb.NodeStructRequest{
		Paths: layout.Displayed,
	})
	if err != nil {
		return fmt.Errorf("cannot query the tree structure: %v", err)
	}
	for _, node := range nodes.Nodes {
		if _, err := dbd.buildPanel(node); err != nil {
			return err
		}
	}
	return nil
}

func (dbd *Dashboard) refresh(data *treepb.NodeData) error {
	pb := data.GetPb()
	if pb == nil {
		return nil
	}
	root := rootpb.RootInfo{}
	if err := pb.UnmarshalTo(&root); err != nil {
		return fmt.Errorf("cannot unmarshal root info: %v", err)
	}
	layout := root.Layout
	if layout == nil {
		return nil
	}
	go func() {
		if err := dbd.updateLayout(layout); err != nil {
			dbd.ui.DisplayErr(err)
		}
	}()
	return nil
}

func (dbd *Dashboard) render(displayData *uipb.DisplayData) {
	for id, nodes := range displayData.Data {
		for _, node := range nodes.Nodes {
			if id == 0 {
				if err := dbd.refresh(node); err != nil {
					dbd.ui.DisplayErr(err)
				}
				continue
			}
			panel := dbd.panels[ui.PanelID(id)]
			if panel == nil {
				continue
			}
			panel.Display(node)
		}
	}
}

// RegisterPanel adds a new panel to display to the dashboard.
func (dbd *Dashboard) RegisterPanel(pnl ui.Panel) error {
	dbd.root.AppendChild(pnl.Root())
	desc := pnl.Desc().(*Descriptor)
	dbd.panels[desc.ID()] = pnl
	return dbd.ui.puller.registerPanel(desc)
}

func (dbd *Dashboard) ClosePanel(pnl ui.Panel) error {
	desc := pnl.Desc().(*Descriptor)
	err := dbd.ui.puller.unregisterPanel(desc)
	delete(dbd.panels, desc.ID())
	dbd.root.RemoveChild(pnl.Root())
	return err
}

func (dbd *Dashboard) buildPanel(node *treepb.Node) (ui.Panel, error) {
	builder := ui.Builder(node.Mime)
	if builder == nil {
		builder = ui.Builder(mime.Unsupported)
	}
	if builder == nil {
		return nil, errors.Errorf("MIME type %q not supported", mime.Unsupported)
	}
	return builder(dbd, node)
}
