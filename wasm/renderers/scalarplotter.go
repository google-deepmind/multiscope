package renderers

import (
	"multiscope/internal/grpc/client"
	"multiscope/internal/style"
	plotpb "multiscope/protos/plot_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	uipb "multiscope/protos/ui_go_proto"
	"syscall/js"
)

type scalarPlot struct {
	gplot *gonumPlot
}

func init() {
	Register(NewScalarPlot)
}

// NewScalarPlot returns a renderer to plot on a canvas using gonum/plot.
func NewScalarPlot(stl *style.Style, panel *uipb.Panel, aux js.Value) Renderer {
	return &scalarPlot{gplot: newGonumPlot(stl, panel, aux)}
}

func (rdr *scalarPlot) Render(data *treepb.NodeData) (*treepb.NodeData, error) {
	pbPlot := &plotpb.ScalarsPlot{}
	if err := client.ToProto(data, pbPlot); err != nil {
		return nil, err
	}
	if pbPlot.Plot == nil {
		return nil, nil
	}
	return rdr.gplot.renderPlot(pbPlot.Plot)
}
