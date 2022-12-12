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
	ui.RegisterBuilderPB(&plotpb.ScalarsPlot{}, newScalarPlot)
}

type scalarPlot struct {
	canvas *dom.HTMLCanvasElement
}

func newScalarPlot(dbd ui.Dashboard, node *treepb.Node) (ui.Panel, error) {
	dsp := &scalarPlot{}
	dsp.canvas = dbd.UI().Owner().Doc().CreateElement("canvas").(*dom.HTMLCanvasElement)
	desc := dbd.NewDescriptor(node, renderers.NewScalarPlot, node.Path)
	desc.AddTransferable("offscreen", dsp.canvas.Call("transferControlToOffscreen"))
	return NewPanel(filepath.Join(node.Path.Path...), desc, dsp)
}

// Display the latest data.
func (dsp *scalarPlot) Display(data *treepb.NodeData) error {
	return nil
}

// Root element of the display.
func (dsp *scalarPlot) Root() dom.HTMLElement {
	return dsp.canvas
}
