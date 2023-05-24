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

package tensor

import (
	"fmt"
	"multiscope/lib/tensor"
	pb "multiscope/protos/tensor_go_proto"
	"reflect"
	"unsafe"
)

type (
	// sTensor is the tensor abstraction used by the server for all data types.
	sTensor interface {
		tensor.Base

		size() int

		ValuesF32() []float32

		Marshal() (*pb.Tensor, error)
	}

	// WithUInt8 is a tensor that can export its content as a uint8 slice.
	WithUInt8 interface {
		ValuesUInt8() []uint8
	}

	// sliceTensor representation used by the server.
	sliceTensor[T tensor.Supported] struct {
		buffer []byte // Keep a reference to the underlying buffer.

		shape   []int
		content []T

		cached []float32
	}

	newer interface {
		newTensor(shape []int, content []byte) sTensor
	}
)

var kindToNewer = map[reflect.Kind]newer{
	reflect.Float32: sliceTensor[float32]{},
	reflect.Float64: sliceTensor[float64]{},
	reflect.Int8:    sliceTensor[int8]{},
	reflect.Int16:   sliceTensor[int16]{},
	reflect.Int32:   sliceTensor[int32]{},
	reflect.Int64:   sliceTensor[int64]{},
	reflect.Uint8:   sliceTensor[uint8]{},
	reflect.Uint16:  sliceTensor[uint16]{},
	reflect.Uint32:  sliceTensor[uint32]{},
	reflect.Uint64:  sliceTensor[uint64]{},
}

func (t *sliceTensor[T]) Marshal() (*pb.Tensor, error) {
	return tensor.Marshal[T](t)
}

func (t *sliceTensor[T]) Shape() []int {
	return t.shape
}

func (t *sliceTensor[T]) Values() []T {
	return t.content
}

func (t *sliceTensor[T]) size() int {
	return len(t.content)
}

func (t *sliceTensor[T]) ValuesF32() []float32 {
	if len(t.content) == 0 {
		return nil
	}
	if t.cached != nil {
		return t.cached
	}
	t.cached = make([]float32, len(t.content))
	for i, v := range t.content {
		t.cached[i] = float32(v)
	}
	return t.cached
}

func (t *sliceTensor[T]) Kind() reflect.Kind {
	var element T
	return reflect.TypeOf(element).Kind()
}

func (t sliceTensor[T]) newTensor(shape []int, buffer []byte) sTensor {
	res := &sliceTensor[T]{
		buffer: buffer,
		shape:  shape,
	}
	if len(buffer) == 0 {
		return res
	}
	first := (*T)(unsafe.Pointer(&buffer[0]))
	res.content = unsafe.Slice(first, tensor.ShapeToLen(shape))
	return res
}

func protoToTensor(pbt *pb.Tensor) (sTensor, error) {
	tp, shape, content, err := tensor.Unmarshal(pbt)
	if err != nil {
		return nil, err
	}
	n, ok := kindToNewer[tp.Kind()]
	if !ok {
		return nil, fmt.Errorf("cannot find a tensor factory for kind %T", tp.Kind())
	}
	return n.newTensor(shape, content), nil
}
