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

package scalar_test

import (
	"testing"

	"multiscope/internal/grpc/grpctesting"
	"multiscope/internal/server/root"
	"multiscope/internal/server/writers/scalar"
	"multiscope/internal/server/writers/scalar/scalartesting"
)

func TestScalarWriter(t *testing.T) {
	rootNode := root.NewRoot()
	state := grpctesting.NewState(rootNode, nil, nil)
	clt, err := grpctesting.SetupTest(state)
	if err != nil {
		t.Fatal(err)
		return
	}
	defer clt.Conn().Close()
	scalarWriter := scalar.NewWriter()
	rootNode.AddChild(scalartesting.Scalar01Name, scalarWriter)
	for _, v := range scalartesting.Scalar01Data {
		if err := scalarWriter.Write(v); err != nil {
			t.Fatal(err)
		}
	}
	if err := scalartesting.CheckScalar01(clt, []string{scalartesting.Scalar01Name}); err != nil {
		t.Error(err)
	}
}
