package renderers

import (
	"image/color"
	"multiscope/internal/grpc/client"
	"multiscope/internal/style"
	"multiscope/internal/wplot"
	plotpb "multiscope/protos/plot_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	uipb "multiscope/protos/ui_go_proto"
	"multiscope/wasm/canvas"
	"syscall/js"

	"github.com/pkg/errors"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
)

type gonumPlot struct {
	canvas *canvas.JSCanvas
	style  *style.Style
}

func init() {
	Register(NewGonumPlot)
}

func newGonumPlot(stl *style.Style, panel *uipb.Panel, aux js.Value) *gonumPlot {
	offscreen := canvas.OffscreenCanvas{Value: aux.Get("offscreen")}
	return &gonumPlot{canvas.New(offscreen), stl}
}

// NewGonumPlot returns a renderer to plot on a canvas using gonum/plot.
func NewGonumPlot(stl *style.Style, panel *uipb.Panel, aux js.Value) Renderer {
	return newGonumPlot(stl, panel, aux)
}

func (rdr *gonumPlot) Render(data *treepb.NodeData) (*treepb.NodeData, error) {
	pbPlot := &plotpb.Plot{}
	if err := client.ToProto(data, pbPlot); err != nil {
		return nil, err
	}
	return rdr.renderPlot(pbPlot)
}

func (rdr *gonumPlot) renderPlot(pbPlot *plotpb.Plot) (*treepb.NodeData, error) {
	rdr.canvas.BeforeDraw()
	defer rdr.canvas.AfterDraw()

	plt := wplot.New()
	plt.SetLineStyles(func(s *draw.LineStyle) {
		s.Color = rdr.style.Foreground()
		s.Width = 2
	})
	plt.SetTextStyles(func(s *draw.TextStyle) {
		s.Color = rdr.style.Foreground()
		s.Font.Size = rdr.style.FontSize()
	})
	theme := rdr.style.Theme()
	colors := []color.Color{
		theme.Color08,
		theme.Color09,
		theme.Color0A,
		theme.Color0B,
		theme.Color0C,
		theme.Color0D,
		theme.Color0E,
		theme.Color0F,
	}
	nextColor := 0
	for _, pbPlotter := range pbPlot.Plotters {
		if pbPlotter == nil || pbPlotter.Serie == nil {
			continue
		}
		plotterColor := colors[nextColor%len(colors)]
		pltr, err := rdr.toPlotter(pbPlotter.Serie, plotterColor)
		if err != nil {
			return nil, errors.Errorf("cannot plot %q: %v", pbPlotter.Legend, err)
		}
		plt.Add(pbPlotter.Legend, pltr)
		nextColor++
	}
	plt.Draw(rdr.canvas.Drawer())
	return nil, nil
}

func (rdr *gonumPlot) toPlotter(serie *plotpb.Serie, plotterColor color.Color) (wplot.Plotter, error) {
	xys := plotter.XYs{}
	for _, point := range serie.Points {
		if point == nil {
			continue
		}
		xys = append(xys, plotter.XY{
			X: point.X,
			Y: point.Y,
		})
	}
	pltr, err := plotter.NewLine(xys)
	if err != nil {
		return nil, err
	}
	pltr.Width = 3
	pltr.Color = plotterColor
	return pltr, nil
}
