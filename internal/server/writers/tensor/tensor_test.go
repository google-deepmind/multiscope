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

package tensor

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestMarshal(t *testing.T) {
	tests := []sTensor{
		&sliceTensor[float32]{
			shape:   []int{1},
			content: []float32{1},
		},
		&sliceTensor[uint8]{
			shape:   []int{1},
			content: []uint8{255},
		},
		&sliceTensor[uint8]{
			shape:   []int{2, 3},
			content: []uint8{1, 2, 3, 4, 5, 6},
		},
	}
	for i, want := range tests {
		pbTensor, err := want.Marshal()
		if err != nil {
			t.Fatalf("test %d: cannot marshal tensor: %v", i, err)
		}
		got, err := protoToTensor(pbTensor)
		if err != nil {
			t.Fatalf("test %d: cannot unmarshal tensor: %v", i, err)
		}
		if !cmp.Equal(got.Shape(), want.Shape()) {
			t.Errorf("test %d: incorrect unserialized tensor shape: got %v but want %v", i, got, want)
		}
		if !cmp.Equal(got.ValuesF32(), want.ValuesF32()) {
			t.Errorf("test %d: incorrect unserialized tensor values: got %v but want %v", i, got, want)
		}
	}
}
