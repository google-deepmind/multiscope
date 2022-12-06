package panels

import (
	plotpb "multiscope/protos/plot_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/renderers"
	"multiscope/wasm/ui"
	"path/filepath"

	"honnef.co/go/js/dom/v2"
)

func init() {
	ui.RegisterBuilderPB(&plotpb.Plot{}, newGonumPlot)
}

type gonumPlot struct {
	canvas *dom.HTMLCanvasElement
}

func newGonumPlot(dbd ui.Dashboard, node *treepb.Node) (ui.Panel, error) {
	dsp := &gonumPlot{}
	dsp.canvas = dbd.UI().Owner().Doc().CreateElement("canvas").(*dom.HTMLCanvasElement)
	dsp.canvas.SetHeight(400)
	dsp.canvas.SetWidth(800)
	desc := dbd.NewDescriptor(node, renderers.NewGonumPlot, node.Path)
	desc.AddTransferable("offscreen", dsp.canvas.Call("transferControlToOffscreen"))
	return NewPanel(filepath.Join(node.Path.Path...), desc, dsp)
}

// Display the latest data.
func (dsp *gonumPlot) Display(data *treepb.NodeData) error {
	return nil
}

// Root element of the display.
func (dsp *gonumPlot) Root() dom.HTMLElement {
	return dsp.canvas
}
