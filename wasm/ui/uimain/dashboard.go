package uimain

import (
	"fmt"
	"multiscope/internal/mime"
	"multiscope/internal/server/core"
	rootpb "multiscope/protos/root_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	uipb "multiscope/protos/ui_go_proto"
	"multiscope/wasm/ui"
	"multiscope/wasm/ui/uimain/dblayout"

	"github.com/pkg/errors"
	"honnef.co/go/js/dom/v2"
)

// Dashboard gets the data sent by the renderers (from a webworker) and display
// them in the main UI thread.
type Dashboard struct {
	ui       *UI
	pathToID map[core.Key]ui.PanelID
	panels   map[ui.PanelID]ui.Panel
	layout   dblayout.Layout
	root     dom.Node
}

func newDashboard(main *UI, rootInfo *rootpb.RootInfo) (*Dashboard, error) {
	const dashboardClass = "container__middle"
	elements := main.Owner().GetElementsByClassName(dashboardClass)
	if len(elements) != 1 {
		return nil, errors.Errorf("wrong number of elements of class %q: got %d but want 1", dashboardClass, len(elements))
	}
	dbd := &Dashboard{
		ui:       main,
		pathToID: make(map[core.Key]ui.PanelID),
		panels:   make(map[ui.PanelID]ui.Panel),
		root:     elements[0],
	}

	var layoutSettings rootpb.Layout
	main.Settings().Listen(dblayout.SettingKey, &layoutSettings, func(src any) error {
		if src == dbd.layout {
			return nil
		}
		return dbd.buildLayout(&layoutSettings)
	})
	if dbd.layout != nil {
		return dbd, nil
	}
	// No layout build from the setting, so we build a new one.
	if err := dbd.buildLayout(rootInfo.Layout); err != nil {
		return nil, err
	}
	return dbd, nil
}

func (dbd *Dashboard) descriptor() *Descriptor {
	return rootDescriptor()
}

// UI returns the UI instance owning the dashboard.
func (dbd *Dashboard) UI() ui.UI {
	return dbd.ui
}

func (dbd *Dashboard) buildLayout(toLoad *rootpb.Layout) error {
	if dbd.layout != nil {
		dbd.clearLayout()
	}

	var err error
	dbd.layout, err = dblayout.New(dbd, toLoad)
	dbd.root.AppendChild(dbd.layout.Root())
	if err != nil {
		return err
	}
	go func() {
		if err := dbd.layout.Load(toLoad); err != nil {
			dbd.ui.DisplayErr(err)
		}
	}()
	return nil
}

func (dbd *Dashboard) refreshFromServer(data *treepb.NodeData) error {
	pb := data.GetPb()
	if pb == nil {
		return nil
	}
	root := rootpb.RootInfo{}
	if err := pb.UnmarshalTo(&root); err != nil {
		return fmt.Errorf("cannot unmarshal root info: %v", err)
	}

	dbd.ui.settings.SetDictKey(root.KeySettings)
	if root.Layout != nil {
		return nil
	}
	return dbd.buildLayout(root.Layout)
}

func (dbd *Dashboard) clearLayout() {
	for _, pnl := range dbd.panels {
		if err := dbd.ClosePanel(pnl); err != nil {
			dbd.ui.DisplayErr(err)
		}
	}
	dbd.root.RemoveChild(dbd.layout.Root())
}

func (dbd *Dashboard) render(displayData *uipb.DisplayData) {
	for id, nodes := range displayData.Data {
		for _, node := range nodes.Nodes {
			if id == 0 {
				if err := dbd.refreshFromServer(node); err != nil {
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

func (dbd *Dashboard) createPanel(node *treepb.Node) error {
	pMime := node.Mime
	if node.Error != "" {
		pMime = mime.Error
	}
	builder := ui.Builder(pMime)
	if builder == nil {
		builder = ui.Builder(mime.Unsupported)
	}
	if builder == nil {
		return errors.Errorf("MIME type %q not supported", mime.Unsupported)
	}
	pnl, err := builder(dbd, node)
	if err != nil {
		return fmt.Errorf("cannot create panel: %w", err)
	}
	return dbd.registerPanel(pnl)
}

// OpenPanel opens a panel on the dashboard or focus on it if the panel has already been opened.
func (dbd *Dashboard) OpenPanel(node *treepb.Node) error {
	id, ok := dbd.pathToID[core.ToKey(node.Path.Path)]
	if ok {
		dbd.panels[id].Root().Call("scrollIntoView")
		return nil
	}
	return dbd.createPanel(node)
}

func (dbd *Dashboard) registerPanel(pnl ui.Panel) error {
	dbd.layout.Append(pnl)
	desc := pnl.Desc().(*Descriptor)
	path := desc.Path()
	if path != nil {
		dbd.pathToID[core.ToKey(path.Path)] = desc.id()
	}
	dbd.panels[desc.id()] = pnl
	return dbd.ui.puller.registerPanel(desc)
}

// ClosePanel closes a panel on the dashboard.
func (dbd *Dashboard) ClosePanel(pnl ui.Panel) error {
	desc := pnl.Desc().(*Descriptor)
	err := dbd.ui.puller.unregisterPanel(desc)
	delete(dbd.panels, desc.id())
	path := desc.Path()
	if path != nil {
		delete(dbd.pathToID, core.ToKey(path.Path))
	}
	dbd.layout.Remove(pnl)
	return err
}
