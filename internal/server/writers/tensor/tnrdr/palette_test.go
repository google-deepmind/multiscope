package tnrdr

import (
	"math"
	"testing"
)

func TestValSampling(t *testing.T) {
	minPos := float32(1000)
	maxPos := float32(0)
	numPos := 0
	minNeg := float32(0)
	maxNeg := float32(-1000)
	numNeg := 0

	for i := 0; i < 256; i++ {
		v := toVal(uint8(i))
		if !math.IsNaN(float64(v)) && v < 0 {
			numNeg++
			if v < minNeg {
				minNeg = v
			}
			if v > maxNeg {
				maxNeg = v
			}
		}
		if !math.IsNaN(float64(v)) && v > 0 {
			numPos++
			if v < minPos {
				minPos = v
			}
			if v > maxPos {
				maxPos = v
			}
		}
	}
	if numPos != numNeg {
		t.Errorf("the number of positive value (%d) is different from the number of negative value (%d)", numPos, numNeg)
	}
	if minPos != 1 {
		t.Errorf("incorrect minimum positive value: got %f but want %f", minPos, 1.)
	}
	if maxPos != 127 {
		t.Errorf("incorrect maximum positive value: got %f but want %f", maxPos, 127.)
	}
	if minNeg != -127 {
		t.Errorf("incorrect minimum negative value: got %f but want %f", minNeg, -127.)
	}
	if maxNeg != -1 {
		t.Errorf("incorrect maximum negative value: got %f but want %f", maxNeg, -1.)
	}
}
