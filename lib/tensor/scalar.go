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
	pb "multiscope/protos/tensor_go_proto"
	"reflect"
	"unsafe"
)

type (
	toScalarAny interface {
		cast([]byte) any
	}

	toScalar[T Supported] struct{}
)

var kindToScalar = map[reflect.Kind]toScalarAny{
	reflect.Float32: toScalar[float32]{},
	reflect.Float64: toScalar[float64]{},
	reflect.Int8:    toScalar[int8]{},
	reflect.Int16:   toScalar[int16]{},
	reflect.Int32:   toScalar[int32]{},
	reflect.Int64:   toScalar[int64]{},
	reflect.Uint8:   toScalar[uint8]{},
	reflect.Uint16:  toScalar[uint16]{},
	reflect.Uint32:  toScalar[uint32]{},
	reflect.Uint64:  toScalar[uint64]{},
}

func (toScalar[T]) cast(content []byte) any {
	var val T
	val = *((*T)(unsafe.Pointer(&content[0])))
	return val
}

// UnmarshalScalar unmarshal a tensor protocol buffer into a scalar.
// The size of content needs to match the size of the scalar.
// The shape of the tensor is ignored.
func UnmarshalScalar(p *pb.Tensor) (any, error) {
	// Create an element of the right type.
	tp, ok := pbToGo[p.Dtype]
	if !ok {
		return nil, fmt.Errorf("cannot unmarshal tensor: type %v is not supported", p.Dtype)
	}
	if len(p.Content) != int(tp.Size()) {
		return nil, fmt.Errorf("wrong content size: got %d bt want the size of a scalar %s (=%d)", len(p.Content), tp, tp.Size())
	}
	toVal, ok := kindToScalar[tp.Kind()]
	if !ok {
		return nil, fmt.Errorf("no converted to scalar for type %s", tp)
	}
	return toVal.cast(p.Content), nil
}
