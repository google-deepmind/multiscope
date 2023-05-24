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

package tnrdr

import (
	"image/color"
	"math"
)

type rgbF struct {
	R, G, B float32
}

const minFac = 4

var (
	maxNeg = rgbF{
		R: 107,
		G: 250,
		B: 255,
	}

	minNeg = rgbF{
		R: maxNeg.R / minFac,
		G: maxNeg.G / minFac,
		B: maxNeg.B / minFac,
	}

	maxPos = rgbF{
		R: 255,
		G: 122,
		B: 77,
	}

	minPos = rgbF{
		R: maxPos.R / minFac,
		G: maxPos.G / minFac,
		B: maxPos.B / minFac,
	}
)

func toColor2(val float32, min, max rgbF) color.RGBA {
	return color.RGBA{
		R: uint8((1-val)*min.R + val*max.R),
		G: uint8((1-val)*min.G + val*max.G),
		B: uint8((1-val)*min.B + val*max.B),
		A: 255,
	}
}

// ToColor returns a color given a metric and a value in the tensor.
func ToColor(m *Metrics, v float32) color.Color {
	if math.IsNaN(float64(v)) {
		return color.RGBA{R: 255, B: 255, A: 255}
	}
	if v == 0 {
		return color.RGBA{A: 255}
	}
	val := (v - m.Min) / m.Range
	if m.Max < 0 {
		return toColor2(val, maxNeg, minNeg)
	}
	if m.Min >= 0 {
		return toColor2(val, minPos, maxPos)
	}
	if v < 0 {
		val = (v - m.Min) / m.AbsMax
		return toColor2(val, maxNeg, minNeg)
	}
	val = v / m.AbsMax
	return toColor2(val, minPos, maxPos)
}
