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

// Package tensor provides helper methods to serialize tensors.
package tensor

import (
	"fmt"
	pb "multiscope/protos/tensor_go_proto"
	"reflect"
	"unsafe"
)

type (
	// Supported is the list of types supported by tensors.
	Supported interface {
		~float64 | ~float32 | ~int64 | ~int32 | ~int16 | ~int8 | ~uint64 | ~uint32 | ~uint16 | ~uint8
	}

	// Base are functions implemented by all tensors.
	Base interface {
		// Shape returns the shape of the Tensor. The product of the dimensions matches the length of the slice returned by Value.
		Shape() []int
	}

	// Tensor defines a float64 tensor.
	Tensor[T Supported] interface {
		Base

		// Value returns a slice stored by the Tensor.
		//
		// The data may not be copied. The slice needs to be copied to own it (e.g. for storage).
		// Not completely sure this is necessary, but it seems safer and does not add any significant cost.
		Values() []T
	}
)

var (
	goToPB = map[reflect.Kind]pb.DataType{
		reflect.Float32: pb.DataType_DT_FLOAT32,
		reflect.Float64: pb.DataType_DT_FLOAT64,
		reflect.Int8:    pb.DataType_DT_INT8,
		reflect.Int16:   pb.DataType_DT_INT16,
		reflect.Int32:   pb.DataType_DT_INT32,
		reflect.Int64:   pb.DataType_DT_INT64,
		reflect.Uint8:   pb.DataType_DT_UINT8,
		reflect.Uint16:  pb.DataType_DT_UINT16,
		reflect.Uint32:  pb.DataType_DT_UINT32,
		reflect.Uint64:  pb.DataType_DT_UINT64,
	}

	pbToGo = map[pb.DataType]reflect.Type{
		pb.DataType_DT_FLOAT32: reflect.TypeOf(float32(0)),
		pb.DataType_DT_FLOAT64: reflect.TypeOf(float64(0)),
		pb.DataType_DT_INT8:    reflect.TypeOf(int8(0)),
		pb.DataType_DT_INT16:   reflect.TypeOf(int16(0)),
		pb.DataType_DT_INT32:   reflect.TypeOf(int32(0)),
		pb.DataType_DT_INT64:   reflect.TypeOf(int64(0)),
		pb.DataType_DT_UINT8:   reflect.TypeOf(uint8(0)),
		pb.DataType_DT_UINT16:  reflect.TypeOf(uint16(0)),
		pb.DataType_DT_UINT32:  reflect.TypeOf(uint32(0)),
		pb.DataType_DT_UINT64:  reflect.TypeOf(uint64(0)),
	}
)

func toPBShape(shp []int) *pb.Shape {
	shape := &pb.Shape{}
	for _, d := range shp {
		shape.Dim = append(shape.Dim, &pb.Shape_Dim{
			Size: int64(d),
		})
	}
	return shape
}

// Marshal a tensor into a Multiscope tensor protocol buffer.
func Marshal[T Supported](t Tensor[T]) (*pb.Tensor, error) {
	shape := toPBShape(t.Shape())
	vals := t.Values()
	var element T
	sizeOf := int(unsafe.Sizeof(element))
	content := make([]byte, len(vals)*sizeOf)
	if len(vals) > 0 {
		// Cast a slice into a byte slice.
		// See https://stackoverflow.com/questions/48756732/what-does-1-30c-yourtype-do-exactly-in-cgo
		buf := (*[1 << 30]byte)(unsafe.Pointer(&(vals[0])))[:len(content):len(content)]
		copy(content, buf)
	}
	dtype, ok := goToPB[reflect.TypeOf(element).Kind()]
	if !ok {
		return nil, fmt.Errorf("internal tnest error: no datatype for %T. Available datatypes are %v", element, goToPB)
	}
	return &pb.Tensor{
		Dtype:   dtype,
		Shape:   shape,
		Content: content,
	}, nil
}

func intShape(shape *pb.Shape) []int {
	dims := shape.GetDim()
	shapeint := make([]int, len(dims))
	for i, dim := range dims {
		shapeint[i] = int(dim.GetSize())
	}
	return shapeint
}

// ShapeToLen return the number of elements given a shape.
func ShapeToLen(shape []int) int {
	l := 0
	if len(shape) > 0 {
		l = 1
	}
	for _, dim := range shape {
		l *= dim
	}
	return l
}

// Unmarshal a protobuf tensor.
func Unmarshal(p *pb.Tensor) (reflect.Type, []int, []byte, error) {
	// Create an element of the right type.
	tp, ok := pbToGo[p.Dtype]
	if !ok {
		return nil, nil, nil, fmt.Errorf("cannot unmarshal tensor: type %v is not supported", p.Dtype)
	}
	// Compute the size of the buffer.
	sizeOf := int(tp.Size())
	if len(p.Content)%sizeOf != 0 {
		return nil, nil, nil, fmt.Errorf("invalid value length: %d is not divisible by the size of a %T (=%d)", len(p.Content), tp, sizeOf)
	}
	shape := intShape(p.Shape)
	length := len(p.Content) / sizeOf
	if len(shape) == 0 && length == 1 {
		// This is a single scalar: adjusts the shape accordingly.
		shape = []int{1}
	}
	shapeLength := ShapeToLen(shape)
	if shapeLength != length {
		return nil, nil, nil, fmt.Errorf(
			"shape %v of length %d does not match the value buffer of length %d=%d/(sizeof(%s)=%d)",
			shape, shapeLength, length, len(p.Content), tp, sizeOf,
		)
	}
	return tp, shape, p.Content, nil
}
