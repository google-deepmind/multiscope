package ui

import (
	tablepb "multiscope/protos/table_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	"multiscope/wasm/renderers"
	"path/filepath"

	"honnef.co/go/js/dom/v2"
)

func init() {
	RegisterDisplayPB(&tablepb.Series{}, newScalarPlot)
}

type scalarPlot struct {
	canvas *dom.HTMLCanvasElement
}

func newScalarPlot(dbd *Dashboard, node *treepb.Node) (*Panel, error) {
	dsp := &scalarPlot{}
	dsp.canvas = dbd.Owner().CreateElement("canvas").(*dom.HTMLCanvasElement)
	dsp.canvas.SetHeight(400)
	dsp.canvas.SetWidth(800)
	desc := NewDescriptor(dsp, dsp.canvas, renderers.NewPlotScalar, node.Path)
	desc.AddTransferable("offscreen", dsp.canvas.Call("transferControlToOffscreen"))
	return dbd.NewPanel(filepath.Join(node.Path.Path...), desc)
}

// Display the latest data.
func (dsp *scalarPlot) Display(data *treepb.NodeData) error {
	return nil
}
