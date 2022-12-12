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
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

type gonumPlot struct {
	canvas *canvas.JSCanvas
	style  *style.Style
}

func init() {
	Register(NewGonumPlot)
}

const (
	defaultRatio     = float32(defaultHeight) / float32(defaultWidth)
	defaultLineWidth = 1
)

func newGonumPlot(stl *style.Style, regPanel *uipb.RegisterPanel, aux js.Value) *gonumPlot {
	offscreen := canvas.OffscreenCanvas{Value: aux.Get("offscreen")}
	width, height := computePreferredSize(regPanel.PreferredSize, defaultRatio)
	offscreen.SetSize(width, height)
	return &gonumPlot{canvas.New(offscreen), stl}
}

// NewGonumPlot returns a renderer to plot on a canvas using gonum/plot.
func NewGonumPlot(stl *style.Style, regPanel *uipb.RegisterPanel, aux js.Value) Renderer {
	return newGonumPlot(stl, regPanel, aux)
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
		s.Width = defaultLineWidth
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
		if pbPlotter == nil {
			continue
		}
		plotterColor := colors[nextColor%len(colors)]
		pltr, err := rdr.toPlotter(pbPlotter, plotterColor)
		if err != nil {
			return nil, errors.Errorf("cannot plot %q: %v", pbPlotter.Legend, err)
		}
		if pltr == nil {
			continue
		}
		plt.Add(pbPlotter.Legend, pltr)
		nextColor++
	}
	plt.Draw(rdr.canvas.Drawer())
	return nil, nil
}

func (rdr *gonumPlot) toPlotter(pbPlotter *plotpb.Plotter, plotterColor color.Color) (pltr wplot.Plotter, err error) {
	if pbPlotter.Drawer == nil {
		return nil, errors.Errorf("no drawer specified")
	}
	switch drw := pbPlotter.Drawer.(type) {
	case *plotpb.Plotter_LineDrawer:
		pltr, err = buildLineDrawer(pbPlotter, drw.LineDrawer, plotterColor)
	case *plotpb.Plotter_HistogramDrawer:
		pltr, err = buildHistogramDrawer(pbPlotter, drw.HistogramDrawer, plotterColor)
	default:
		return nil, errors.Errorf("drawer %T not supported", drw)
	}
	return
}

func buildLineDrawer(pbPlotter *plotpb.Plotter, drw *plotpb.LineDrawer, plotterColor color.Color) (wplot.Plotter, error) {
	if len(drw.Points) == 0 {
		return nil, nil
	}
	xys := plotter.XYs{}
	for _, point := range drw.Points {
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
	pltr.Width = defaultLineWidth
	pltr.Color = plotterColor
	return pltr, nil
}

var histogramLineStyle = draw.LineStyle{
	Color:    color.Black,
	Width:    vg.Points(0),
	Dashes:   []vg.Length{},
	DashOffs: 0,
}

func buildHistogramDrawer(pbPlotter *plotpb.Plotter, drw *plotpb.HistogramDrawer, plotterColor color.Color) (wplot.Plotter, error) {
	if len(drw.Bins) == 0 {
		return nil, nil
	}
	n := len(drw.Bins)
	bins := make([]plotter.HistogramBin, n)
	for i, pbBin := range drw.Bins {
		bins[i].Min = pbBin.Min
		bins[i].Max = pbBin.Max
		bins[i].Weight = pbBin.Weight
	}
	xmin, xmax := bins[0].Min, bins[n-1].Max
	width := (xmax - xmin) / float64(n)
	if width == 0 {
		width = 1
	}
	return &plotter.Histogram{
		Bins:      bins,
		Width:     width,
		FillColor: plotterColor,
		LineStyle: histogramLineStyle,
	}, nil
}
