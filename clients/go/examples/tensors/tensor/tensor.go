// Package tensor declares a minimal tensor implementation for the examples.
package tensor

import "multiscope/clients/go/remote"

// Tensor implements the Multiscope tensor interface.
type Tensor struct {
	shape []int
	vals  []float32
}

var _ remote.Tensor = (*Tensor)(nil)

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
