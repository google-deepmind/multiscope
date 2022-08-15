package ui

import (
	"context"
	"fmt"
	rootpb "multiscope/protos/root_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	uipb "multiscope/protos/ui_go_proto"

	"github.com/pkg/errors"
	"honnef.co/go/js/dom/v2"
)

// Dashboard gets the data sent by the renderers (from a webworker) and display
// them in the main UI thread.
type Dashboard struct {
	ui     *UI
	panels map[PanelID]*Panel
	root   dom.Node
}

func newDashboard(ui *UI) (*Dashboard, error) {
	const dashboardClass = "container__middle"
	elements := ui.Owner().GetElementsByClassName(dashboardClass)
	if len(elements) != 1 {
		return nil, errors.Errorf("wrong number of elements of class %q: got %d but want 1", dashboardClass, len(elements))
	}
	return &Dashboard{
		ui:     ui,
		panels: make(map[PanelID]*Panel),
		root:   elements[0],
	}, nil
}

func (dbd *Dashboard) descriptor() *Descriptor {
	return rootDescriptor()
}

// NewErrorElement returns an element to display an error.
func (dbd *Dashboard) NewErrorElement() dom.HTMLElement {
	el := dbd.ui.Owner().CreateElement("p").(dom.HTMLElement)
	el.Class().Add("panel-error")
	return el
}

// Owner returns the owner of the DOM tree of the UI.
func (dbd *Dashboard) Owner() dom.HTMLDocument {
	return dbd.ui.Owner()
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
			panel := dbd.panels[PanelID(id)]
			if panel == nil {
				continue
			}
			panel.Display(node)
		}
	}
}
