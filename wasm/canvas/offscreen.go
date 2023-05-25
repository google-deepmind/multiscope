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

// SetSize sets the size of the canvas.
func (oc OffscreenCanvas) SetSize(width, height int) {
	oc.Set("width", width)
	oc.Set("height", height)
}

// Width returns the width of the canvas.
func (oc OffscreenCanvas) Width() int {
	return oc.Get("width").Int()
}

// Height returns the height of the canvas.
func (oc OffscreenCanvas) Height() int {
	return oc.Get("height").Int()
}
