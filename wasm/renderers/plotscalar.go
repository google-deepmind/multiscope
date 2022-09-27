package renderers

import (
	"image/color"
	"multiscope/internal/grpc/client"
	"multiscope/internal/style"
	"multiscope/internal/wplot"
	tablepb "multiscope/protos/table_go_proto"
	treepb "multiscope/protos/tree_go_proto"
	uipb "multiscope/protos/ui_go_proto"
	"multiscope/wasm/canvas"
	"sort"
	"syscall/js"

	"github.com/pkg/errors"
	"go.uber.org/multierr"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
)

type plotScalar struct {
	canvas *canvas.JSCanvas
	style  *style.Style
}

func init() {
	Register(NewPlotScalar)
}

// NewPlotScalar returns a renderer to plot scalars on a canvas using gonum/plot.
func NewPlotScalar(stl *style.Style, panel *uipb.Panel, aux js.Value) Renderer {
	offscreen := canvas.OffscreenCanvas{Value: aux.Get("offscreen")}
	return &plotScalar{canvas.New(offscreen), stl}
}

func labels(m map[string]*tablepb.Serie) []string {
	lbls := []string{}
	for lbl := range m {
		lbls = append(lbls, lbl)
	}
	sort.Strings(lbls)
	return lbls
}

func (rdr *plotScalar) Render(data *treepb.NodeData) (*treepb.NodeData, error) {
	rdr.canvas.BeforeDraw()
	defer rdr.canvas.AfterDraw()

	series := &tablepb.Series{}
	if err := client.ToProto(data, series); err != nil {
		return nil, err
	}

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
	var gErr error
	for _, label := range labels(series.LabelToSerie) {
		serie := series.LabelToSerie[label]
		xys := make(plotter.XYs, len(serie.Points))
		for i, point := range serie.Points {
			xys[i].X = point.X
			xys[i].Y = point.Y
		}
		l, err := plotter.NewLine(xys)
		l.Color = colors[nextColor%len(colors)]
		l.Width = 3
		if err != nil {
			gErr = multierr.Append(gErr, errors.Errorf("error while adding the line to the plot: %v", err))
		}
		plt.Add(label, l)
		nextColor++
	}
	plt.Draw(rdr.canvas.Drawer())
	return nil, gErr
}
