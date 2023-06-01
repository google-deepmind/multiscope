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

package scope_test

import (
	"testing"

	"multiscope/clients/go/clienttesting"
	"multiscope/clients/go/remote"
	"multiscope/internal/server/writers/tensor"
	tensortesting "multiscope/internal/server/writers/tensor/testing"
)

type tensorS struct {
	shape []int64
	value []float32
}

func (t *tensorS) Shape() []int64 {
	return t.shape
}

func (t *tensorS) Value() []float32 {
	return t.value
}

func TestTensorWriter(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	writer, err := remote.NewTensorWriter[float32](clt, tensortesting.Tensor01Name, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := clienttesting.ForceActive(clt.TreeClient(), writer.Path()); err != nil {
		t.Fatal(err)
	}
	for _, test := range tensortesting.TensorTests {
		if err := writer.Write(test.Tensor); err != nil {
			t.Fatal(err)
		}
		if err := tensortesting.CheckTensorData(clt.TreeClient(), writer.Path(), &test); err != nil {
			t.Errorf("incorrect tensor data for test %q: %v", test.Desc, err)
		}
	}
}

func TestTensorWriterForwardImageAndShouldWrite(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	writer, err := remote.NewTensorWriter[float32](clt, tensortesting.Tensor01Name, nil)
	if err != nil {
		t.Fatal(err)
	}
	path := []string{tensortesting.Tensor01Name, tensor.NodeNameImage}
	if err := clienttesting.CheckBecomeActive(clt, path, writer); err != nil {
		t.Error(err)
	}
}

func TestTensorWriterForwardDistributionAndShouldWrite(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	writer, err := remote.NewTensorWriter[float32](clt, tensortesting.Tensor01Name, nil)
	if err != nil {
		t.Fatal(err)
	}
	path := []string{tensortesting.Tensor01Name, tensor.NodeNameDistribution}
	if err := clienttesting.CheckBecomeActive(clt, path, writer); err != nil {
		t.Error(err)
	}
}

func TestTensorWriterForwardMinMaxAndShouldWrite(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	writer, err := remote.NewTensorWriter[float32](clt, tensortesting.Tensor01Name, nil)
	if err != nil {
		t.Fatal(err)
	}
	path := []string{tensortesting.Tensor01Name, tensor.NodeNameMinMax}
	if err := clienttesting.CheckBecomeActive(clt, path, writer); err != nil {
		t.Error(err)
	}
}

func TestTensorWriterForwardNormsAndShouldWrite(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	writer, err := remote.NewTensorWriter[float32](clt, tensortesting.Tensor01Name, nil)
	if err != nil {
		t.Fatal(err)
	}
	path := []string{tensortesting.Tensor01Name, tensor.NodeNameNorms}
	if err := clienttesting.CheckBecomeActive(clt, path, writer); err != nil {
		t.Error(err)
	}
}
