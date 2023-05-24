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

// Package tensor declares a minimal tensor implementation for the examples.
package tensor

import "multiscope/lib/tensor"

// Tensor implements the Multiscope tensor interface.
type Tensor struct {
	shape []int
	vals  []float32
}

var _ tensor.Tensor[float32] = (*Tensor)(nil)

// NewTensor creates a new tensor given a shape.
func NewTensor(shape ...int) *Tensor {
	l := 1
	for _, s := range shape {
		l *= s
	}
	return &Tensor{
		shape: shape,
		vals:  make([]float32, l),
	}
}

// Shape returns the shape of the tensor.
func (t *Tensor) Shape() []int {
	return t.shape
}

// Values returns a slice of all the values in the tensor.
// The slice can be modified in place to change the value of the tensor.
func (t *Tensor) Values() []float32 {
	return t.vals
}
