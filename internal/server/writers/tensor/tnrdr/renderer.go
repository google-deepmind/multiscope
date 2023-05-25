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

// Package tnrdr renders tensors into images.
package tnrdr

import (
	"image"
	"image/color"
	"math"
)

type (
	// Image that can have its pixels set.
	Image interface {
		image.Image
		Set(x, y int, c color.Color)
	}

	// Tensor defines a float64 tensor.
	Tensor interface {
		// Shape returns the shape of the Tensor.
		Shape() []int

		// Values returns the tensor values.
		ValuesF32() []float32
	}

	// Renderer renders a tensor into an image.
	Renderer struct {
		imgF func(image.Rectangle) Image
	}
)

// NewRGBA is a RGBA image factory.
func NewRGBA(r image.Rectangle) Image {
	return image.NewRGBA(r)
}

// NewRenderer returns a renderer given an image factory.
func NewRenderer(imgF func(image.Rectangle) Image) *Renderer {
	return &Renderer{imgF: imgF}
}

func numPix(img Image) int {
	bounds := img.Bounds()
	return bounds.Dx() * bounds.Dy()
}

// ClearImage returns an empty image (to represent empty tensors).
func (r Renderer) ClearImage(img Image) Image {
	if img != nil && numPix(img) == 0 {
		return img
	}
	return r.imgF(image.Rect(0, 0, 0, 0))
}

func squeezedDims(shape []int) int {
	s := 0
	for _, d := range shape {
		if d > 1 {
			s++
		}
	}
	return s
}

// Render a tensor into an image.
func (r Renderer) Render(t Tensor, m *Metrics, img Image) Image {
	if len(t.ValuesF32()) == 0 {
		return r.ClearImage(img)
	}
	if squeezedDims(t.Shape()) <= 1 {
		return r.render1DTensor(t, m, img)
	}
	return r.renderNDTensor(t, m, img)
}

func (r Renderer) resize1DImage(t Tensor, img Image) Image {
	if img != nil && numPix(img) == len(t.ValuesF32()) {
		return img
	}
	sqrt := math.Sqrt(float64(len(t.ValuesF32())))
	dimX := int(math.Ceil(sqrt))
	dimY := int(math.Floor(sqrt))
	for ; dimY*dimX < len(t.ValuesF32()); dimY++ {
	}
	return r.imgF(image.Rect(0, 0, dimX, dimY))
}

func (r Renderer) render1DTensor(t Tensor, m *Metrics, img Image) Image {
	img = r.resize1DImage(t, img)
	bounds := img.Bounds()
	for i, v := range t.ValuesF32() {
		x := i % bounds.Dx()
		y := i / bounds.Dx()
		img.Set(x, y, ToColor(m, v))
	}
	return img
}

func (r Renderer) resizeNDImage(t Tensor, img Image) Image {
	dimY := 1
	dimX := 1
	for i, dimI := range t.Shape() {
		if i%2 == 0 {
			dimY *= dimI
		} else {
			dimX *= dimI
		}
	}
	if img != nil && img.Bounds().Dx() == dimX && img.Bounds().Dy() == dimY {
		return img
	}
	return r.imgF(image.Rect(0, 0, dimX, dimY))
}

type renderContext struct {
	t                Tensor
	m                *Metrics
	img              Image
	workingShape     []int
	offset           int
	stride           int
	dimX, dimY       int
	imgPosX, imgPosY int
	horizontal       bool
}

func renderImg(ctx renderContext) {
	for i := 0; i < ctx.dimX*ctx.dimY; i++ {
		pos := ctx.offset + ctx.stride*i
		x := i % ctx.dimX
		y := i / ctx.dimX
		rgba := ToColor(ctx.m, ctx.t.ValuesF32()[pos])
		ctx.img.Set(ctx.imgPosX+x, ctx.imgPosY+y, rgba)
	}
}

func computeOffsetIncrement(shape []int) int {
	s := 1
	for _, d := range shape {
		s *= d
	}
	return s
}

func renderRecursive(ctx renderContext) {
	if len(ctx.workingShape) == 2 {
		renderImg(ctx)
		return
	}

	cur := ctx
	dim := len(cur.workingShape) - 1
	size := cur.workingShape[dim]
	cur.stride *= size
	cur.workingShape = cur.workingShape[:dim]
	cur.horizontal = len(cur.workingShape)%2 == 0
	offsetInc := computeOffsetIncrement(cur.t.Shape()[dim+1:])
	for i := 0; i < size; i++ {
		cur.offset = ctx.offset + i*offsetInc
		if ctx.horizontal {
			cur.imgPosX = ctx.imgPosX + i*ctx.dimX
		} else {
			cur.imgPosY = ctx.imgPosY + i*ctx.dimY
		}
		renderRecursive(cur)
	}
}

func (r Renderer) renderNDTensor(t Tensor, m *Metrics, img Image) Image {
	img = r.resizeNDImage(t, img)
	renderRecursive(renderContext{
		t:            t,
		m:            m,
		img:          img,
		workingShape: t.Shape(),
		horizontal:   len(t.Shape())%2 == 0,
		offset:       0,
		stride:       1,
		dimY:         t.Shape()[0],
		dimX:         t.Shape()[1],
	})
	return img
}
