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

//go:build js

// Package canvas implements a gonum canvas using a Javascript canvas.
package canvas

import (
	"fmt"
	"image"
	"image/color"

	"multiscope/internal/css"
	"multiscope/internal/wplot"

	"gonum.org/v1/plot/font"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"honnef.co/go/js/dom/v2"
)

type (
	// Element is the owner of a 2D canvas rendering context.
	Element interface {
		GetContext2d() *dom.CanvasRenderingContext2D

		Width() int

		Height() int
	}

	// JSCanvas is a gonum/plot canvas drawer using a Javascript canvas as a backend.
	JSCanvas struct {
		width, height vg.Length
		element       Element
		ctx           *dom.CanvasRenderingContext2D
	}
)

// Check that the Element interface is compatible with a HTML canvas.
var _ Element = (*dom.HTMLCanvasElement)(nil)

func toPix(l vg.Length) float64 {
	return wplot.ToPix(l)
}

// New returns a new canvas using the same context.
func New(element Element) *JSCanvas {
	c := &JSCanvas{
		element: element,
		ctx:     element.GetContext2d(),
	}
	return c
}

func (c *JSCanvas) updateSize() {
	c.width = wplot.ToLength(c.element.Width())
	c.height = wplot.ToLength(c.element.Height())
}

// Size returns the size of the canvas.
func (c *JSCanvas) Size() (x, y vg.Length) {
	return c.width, c.height
}

// SetLineWidth sets the width of stroked paths.
// If the width is not positive then stroked lines
// are not drawn.
//
// The initial line width is 1 point.
func (c *JSCanvas) SetLineWidth(l vg.Length) {
	c.ctx.SetLineWidth(int(wplot.ToPix(l)))
}

// SetLineDash sets the dash pattern for lines.
// The pattern slice specifies the lengths of
// alternating dashes and gaps, and the offset
// specifies the distance into the dash pattern
// to start the dash.
//
// The initial dash pattern is a solid line.
func (c *JSCanvas) SetLineDash(pattern []vg.Length, offset vg.Length) {
}

// SetColor sets the current drawing color.
// Note that fill color and stroke color are
// the same, so if you want different fill
// and stroke colors then you must set a color,
// draw fills, set a new color and then draw lines.
//
// The initial color is black.
// If SetColor is called with a nil color then black is used.
func (c *JSCanvas) SetColor(clr color.Color) {
	style := css.Color(clr)
	c.ctx.SetStrokeStyle(style)
	c.ctx.SetFillStyle(style)
}

// Rotate applies a rotation transform to the context.
// The parameter is specified in radians.
func (c *JSCanvas) Rotate(rad float64) {
	fmt.Println("Rotate", rad)
}

// Translate applies a translational transform
// to the context.
func (c *JSCanvas) Translate(pt vg.Point) {
	c.ctx.Translate(toPix(pt.X), toPix(pt.Y))
}

// Scale applies a scaling transform to the
// context.
func (c *JSCanvas) Scale(x, y float64) {
	fmt.Println("Scale", x, y)
}

// Push saves the current line width, the
// current dash pattern, the current
// transforms, and the current color
// onto a stack so that the state can later
// be restored by calling Pop().
func (c *JSCanvas) Push() {
	fmt.Println("Push")
}

// Pop restores the context saved by the
// corresponding call to Push().
func (c *JSCanvas) Pop() {
	fmt.Println("Pop")
}

// Stroke strokes the given path.
func (c *JSCanvas) Stroke(path vg.Path) {
	if len(path) == 0 {
		return
	}
	c.ctx.BeginPath()
	defer c.ctx.ClosePath()
	c.ctx.MoveTo(toPix(path[0].Pos.X), toPix(c.height-path[0].Pos.Y))
	for _, p := range path[1:] {
		c.ctx.LineTo(toPix(p.Pos.X), toPix(c.height-p.Pos.Y))
	}
	c.ctx.Stroke()
}

// Fill fills the given path.
func (c *JSCanvas) Fill(path vg.Path) {
	if len(path) == 0 {
		return
	}
	c.ctx.BeginPath()
	defer c.ctx.ClosePath()
	c.ctx.MoveTo(toPix(path[0].Pos.X), toPix(c.height-path[0].Pos.Y))
	for _, p := range path[1 : len(path)-1] {
		c.ctx.LineTo(toPix(p.Pos.X), toPix(c.height-p.Pos.Y))
	}
	c.ctx.Fill()
}

// FillString fills in text at the specified location using the given font.
// If the font size is zero, the text is not drawn.
func (c *JSCanvas) FillString(f font.Face, pt vg.Point, text string) {
	c.ctx.SetFont(fmt.Sprintf("%.0fpx %s", f.Font.Size.Points(), f.Name()))
	c.ctx.FillText(text, toPix(pt.X), toPix(c.height-pt.Y), -1)
}

// DrawImage draws the image, scaled to fit
// the destination rectangle.
func (c *JSCanvas) DrawImage(rect vg.Rectangle, img image.Image) {
	fmt.Println("DrawImage", rect, img)
}

// BeforeDraw prepares the canvas for drawing a frame.
func (c *JSCanvas) BeforeDraw() {
	c.ctx.Save()
	c.updateSize()
	c.ctx.ClearRect(0, 0, toPix(c.width), toPix(c.height))
}

// AfterDraw finishes drawing the current frame.
func (c *JSCanvas) AfterDraw() {
	c.ctx.Restore()
}

// Drawer returns the gonum/plot drawer.
func (c *JSCanvas) Drawer() draw.Canvas {
	return draw.New(c)
}
