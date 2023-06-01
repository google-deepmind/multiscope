// Copyright 2023 DeepMind Technologies Limited
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
