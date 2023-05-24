// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package wplot provides gonum plots on the frontend.
package wplot

import (
	"image/color"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/font"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

type (
	// Margins around a plot.
	Margins struct {
		// Margins between the plot and the image border.
		// Must a ratio between 0 and 1.
		Left, Right, Bottom, Top float64
	}

	// PlotterMaker is an action, like drawing a line, on the plot when data is added.
	PlotterMaker interface {
		Make(g *WPlot, label string, xys plotter.XYs) (plot.Thumbnailer, error)
	}

	// WPlot represents a plot with margins and different defaults compared to gonum/plot.
	WPlot struct {
		*plot.Plot
		Margins

		hasLegend bool
	}
)

func defaultSetStyle(style *draw.TextStyle) {
	style.Font.Typeface = "Arial"
	style.Font.Variant = "Sans"
	style.Font.Size = 32
}

// DefaultMargin is the margin to use around a plot (in proportion of the total image size).
var DefaultMargin = 0.03

// New returns a new plot.
func New() *WPlot {
	g := &WPlot{
		Plot: plot.New(),
		Margins: Margins{
			Left:   DefaultMargin,
			Right:  DefaultMargin,
			Bottom: DefaultMargin,
			Top:    DefaultMargin,
		},
	}
	g.BackgroundColor = color.Transparent
	g.SetTextStyles(defaultSetStyle)
	return g
}

// ToLengthF converts a number of pixels into a length.
func ToLengthF(pixels float64) vg.Length {
	return font.Points(pixels * .75)
}

// ToLength converts a number of pixels into a length.
func ToLength(pixels int) vg.Length {
	return ToLengthF(float64(pixels))
}

// ToPix converts a length into a number of pixels.
func ToPix(l vg.Length) float64 {
	return float64(l) / .75
}

// SetTextStyles calls a callback function to set the style of all the text used in a plot.
func (g *WPlot) SetTextStyles(set func(*draw.TextStyle)) {
	styles := []*draw.TextStyle{
		&g.X.Tick.Label,
		&g.X.Label.TextStyle,
		&g.Y.Tick.Label,
		&g.Y.Label.TextStyle,
		&g.Legend.TextStyle,
		&g.Title.TextStyle,
	}
	for _, style := range styles {
		set(style)
	}
}

// SetLineStyles calls a callback function to set the style of all the lines used in a plot
// (excluding the data itself).
func (g *WPlot) SetLineStyles(set func(*draw.LineStyle)) {
	styles := []*draw.LineStyle{
		&g.Plot.X.LineStyle,
		&g.Plot.Y.Tick.LineStyle,
		&g.Plot.Y.LineStyle,
		&g.Plot.X.Tick.LineStyle,
	}
	for _, style := range styles {
		set(style)
	}
}

// SetAxis applies axis options on the axis.
func (g *WPlot) SetAxis(x, y []AxisOption) {
	applyAxisOptions(&g.X, x)
	applyAxisOptions(&g.Y, y)
}

func cropMargins(mrgn Margins, cnvs draw.Canvas) draw.Canvas {
	w := float64(cnvs.Max.X - cnvs.Min.X)
	h := float64(cnvs.Max.Y - cnvs.Min.Y)
	left := vg.Length(mrgn.Left * w)
	right := vg.Length(-mrgn.Right * w)
	bottom := vg.Length(mrgn.Bottom * h)
	top := vg.Length(-mrgn.Top * h)
	return draw.Crop(cnvs, left, right, bottom, top)
}

// Plotter plots data that can be displayed in a legend.
type Plotter interface {
	plot.Plotter
	plot.Thumbnailer
}

// Add plotter to the plot.
func (g *WPlot) Add(label string, p Plotter) {
	if label != "" {
		g.hasLegend = true
		g.Legend.Add(label, p)
	}
	g.Plot.Add(p)
}

// Draw the plot on a HTML canvas.
func (g *WPlot) Draw(cnvs draw.Canvas) {
	cropped := cropMargins(g.Margins, cnvs)
	g.Plot.Draw(cropped)
}
