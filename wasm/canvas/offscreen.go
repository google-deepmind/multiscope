//go:build js

package canvas

import (
	"syscall/js"

	"honnef.co/go/js/dom/v2"
)

// OffscreenCanvas is an offscreen canvas.
type OffscreenCanvas struct {
	js.Value
}

var _ Element = (*OffscreenCanvas)(nil)

// GetContext2d returns the 2d context to draw off screen.
func (oc OffscreenCanvas) GetContext2d() *dom.CanvasRenderingContext2D {
	return &dom.CanvasRenderingContext2D{Value: oc.Call("getContext", "2d")}
}

// SetWidth sets the width of the canvas.
func (oc OffscreenCanvas) SetWidth(w int) {
	oc.Set("width", w)
}

// SetHeight sets the height of the canvas.
func (oc OffscreenCanvas) SetHeight(h int) {
	oc.Set("height", h)
}

// Width returns the width of the canvas.
func (oc OffscreenCanvas) Width() int {
	return oc.Get("width").Int()
}

// Height returns the height of the canvas.
func (oc OffscreenCanvas) Height() int {
	return oc.Get("height").Int()
}
