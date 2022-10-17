package tnrdr

import (
	"math"
)

// Metrics keeps statistics about tensors.
type Metrics struct {
	HistMin, HistMax float32
	Min, Max         float32
	AbsMax           float32
	L1Norm, L2Norm   float32
	Range            float32
}

// Reset all metrics.
func (m *Metrics) Reset() {
	m.HistMax = -math.MaxFloat32
	m.HistMin = math.MaxFloat32
	m.Max = -math.MaxFloat32
	m.Min = math.MaxFloat32
	m.L1Norm = 0
	m.L2Norm = 0
	m.Range = 0
	m.AbsMax = 0
}

func max(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func abs(a float32) float32 {
	if a < 0 {
		return -a
	}
	return a
}

// Update all metrics given a tensor.
func (m *Metrics) Update(values []float32) {
	m.Max = -math.MaxFloat32
	m.Min = math.MaxFloat32
	m.L1Norm = 0
	m.L2Norm = 0
	if len(values) == 0 {
		return
	}
	for _, v := range values {
		// Min and max
		if v > m.Max {
			m.Max = v
		}
		if v < m.Min {
			m.Min = v
		}
		// L2Norm
		m.L2Norm += v * v
		// L1Norm
		if v > 0 {
			m.L1Norm += v
		} else {
			m.L1Norm += -v
		}
	}
	m.Range = m.Max - m.Min
	m.AbsMax = max(abs(m.Min), abs(m.Max))
	m.L2Norm = float32(math.Sqrt(float64(m.L2Norm)))
	if m.Max > m.HistMax {
		m.HistMax = m.Max
	}
	if m.Min < m.HistMin {
		m.HistMin = m.Min
	}
}
