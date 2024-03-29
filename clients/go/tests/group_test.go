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
	"fmt"
	"testing"

	"multiscope/clients/go/clienttesting"
	"multiscope/clients/go/remote"
	"multiscope/internal/server/writers/scalar/scalartesting"
	"multiscope/internal/server/writers/text/texttesting"

	"github.com/google/go-cmp/cmp"
)

const groupName = "Parent"

func writeAndCheckScalarTextWriters(clt *remote.Client, prefix remote.Path, scalarWriter *remote.ScalarWriter, textWriter *remote.TextWriter) error {
	wantScalarPath := prefix.Append(scalartesting.Scalar01Name)
	gotScalarPath := scalarWriter.Path()
	if !cmp.Equal(gotScalarPath, wantScalarPath) {
		return fmt.Errorf("incorrect path: got %v but want %v", gotScalarPath, wantScalarPath)
	}
	// Write and test the data.
	for i, data := range scalartesting.Scalar01Data {
		if err := textWriter.Write(texttesting.Text01Data[i]); err != nil {
			return err
		}
		if err := texttesting.CheckText01(clt, textWriter.Path(), i); err != nil {
			return err
		}
		if err := scalarWriter.WriteFloat64(data); err != nil {
			return err
		}
	}
	return scalartesting.CheckScalar01(clt, scalarWriter.Path())
}

func TestScalarTextWritersWithGroup(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Error(err)
	}
	parent, err := remote.NewGroup(clt, groupName, nil)
	if err != nil {
		t.Fatal(err)
	}
	scalarWriter, err := remote.NewScalarWriter(clt, scalartesting.Scalar01Name, parent.Path())
	if err != nil {
		t.Fatal(err)
	}
	textWriter, err := remote.NewTextWriter(clt, texttesting.Text01Name, parent.Path())
	if err != nil {
		t.Fatal(err)
	}
	if err := writeAndCheckScalarTextWriters(clt, parent.Path(), scalarWriter, textWriter); err != nil {
		t.Errorf("error when creating the writer under a group: %v", err)
	}
}

func TestScalarTextWritersWithChildClient(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Error(err)
	}
	if clt, err = clt.NewChildClient(groupName); err != nil {
		t.Fatal(err)
	}
	scalarWriter, err := remote.NewScalarWriter(clt, scalartesting.Scalar01Name, nil)
	if err != nil {
		t.Fatal(err)
	}
	textWriter, err := remote.NewTextWriter(clt, texttesting.Text01Name, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := writeAndCheckScalarTextWriters(clt, clt.Prefix(), scalarWriter, textWriter); err != nil {
		t.Errorf("error when creating the writer with a child client: %v", err)
	}
}

func TestSuccessiveCreationWithChildClient(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Error(err)
	}
	if clt, err = clt.NewChildClient("Parent 01"); err != nil {
		t.Fatal(err)
	}
	parent, err := remote.NewGroup(clt, "Parent 02", nil)
	if err != nil {
		t.Fatal(err)
	}
	parent, err = remote.NewGroup(clt, groupName, parent.Path())
	if err != nil {
		t.Fatal(err)
	}
	scalarWriter, err := remote.NewScalarWriter(clt, scalartesting.Scalar01Name, parent.Path())
	if err != nil {
		t.Fatal(err)
	}
	textWriter, err := remote.NewTextWriter(clt, texttesting.Text01Name, parent.Path())
	if err != nil {
		t.Fatal(err)
	}
	if err := writeAndCheckScalarTextWriters(clt, parent.Path(), scalarWriter, textWriter); err != nil {
		t.Errorf("error when creating the writer with multiple parents: %v", err)
	}
}
