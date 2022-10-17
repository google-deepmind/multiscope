package tnrdr

import (
	"image/color"
	"math"
)

func toVal(i uint8) float32 {
	// First value is 0.
	if i == 0 {
		return 0
	}
	// Last value is NaN.
	if i == 255 {
		return float32(math.NaN())
	}
	// First half values are positive.
	if i < 128 {
		return float32(i)
	}
	// Second half values are negative.
	return -(256 - float32(i)) + 1
}

// BuildPalette returns a palette with colors adapted to render a tensor.
func BuildPalette() []color.Color {
	vals := make([]float32, 256)
	for i := range vals {
		vals[i] = toVal(uint8(i))
	}

	m := &Metrics{}
	m.Update(vals)

	palette := make([]color.Color, 256)
	for i, val := range vals {
		palette[i] = ToColor(m, val)
	}
	return palette
}
