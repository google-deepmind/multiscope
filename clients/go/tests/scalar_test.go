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
	"multiscope/internal/server/writers/scalar/scalartesting"
)

func TestScalarWriter(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	scalarWriter, err := remote.NewScalarWriter(clt, scalartesting.Scalar01Name, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, data := range scalartesting.Scalar01Data {
		if err := scalarWriter.WriteFloat64(data); err != nil {
			t.Fatal(err)
		}
	}
	if err := scalartesting.CheckScalar01(clt.TreeClient(), scalarWriter.Path()); err != nil {
		t.Error(err)
	}
}
