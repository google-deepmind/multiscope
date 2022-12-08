package panels

import (
	"multiscope/internal/mime"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/renderers"
	"multiscope/wasm/ui"
	"path/filepath"

	"honnef.co/go/js/dom/v2"
)

func init() {
	ui.RegisterBuilder(mime.PNG, newImagePanel)
}

type imagePanel struct {
	canvas *dom.HTMLCanvasElement
}

func newImagePanel(dbd ui.Dashboard, node *treepb.Node) (ui.Panel, error) {
	dsp := &imagePanel{}
	dsp.canvas = dbd.UI().Owner().Doc().CreateElement("canvas").(*dom.HTMLCanvasElement)
	desc := dbd.NewDescriptor(node, renderers.NewImageRenderer, node.Path)
	desc.AddTransferable("offscreen", dsp.canvas.Call("transferControlToOffscreen"))
	pnl, err := NewPanel(filepath.Join(node.Path.Path...), desc, dsp)
	if err != nil {
		return nil, err
	}
	pnl.OnResize(dsp.onResize)
	return pnl, nil
}

func (dsp *imagePanel) onResize(pnl *Panel) {
	dsp.canvas.SetWidth(pnl.width())
	dsp.canvas.SetHeight(pnl.height())
}

// Display the latest data.
func (dsp *imagePanel) Display(data *treepb.NodeData) error {
	return nil
}

// Root element of the display.
func (dsp *imagePanel) Root() dom.HTMLElement {
	return dsp.canvas
}
