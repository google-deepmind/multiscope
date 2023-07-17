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
	"text/template"

	"multiscope/clients/go/clienttesting"
	"multiscope/clients/go/remote"
	"multiscope/internal/server/writers/text/texttesting"
)

func TestWriter(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	writer, err := remote.NewTextWriter(clt, texttesting.Text01Name, nil)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range texttesting.Text01Data {
		if err := writer.Write(want); err != nil {
			t.Error(err)
			break
		}
		if err := texttesting.CheckText01(clt, []string{texttesting.Text01Name}, i); err != nil {
			t.Error(err)
		}
	}
}

func TestWriterIO(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	writer, err := remote.NewTextWriter(clt, texttesting.Text01Name, nil)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := template.Must(template.New("").Parse("{{.content}}"))
	for i, want := range texttesting.Text01Data {
		if err := tmpl.Execute(writer.IO(), map[string]string{
			"content": want,
		}); err != nil {
			t.Error(err)
			break
		}
		if err := writer.Write(want); err != nil {
			t.Error(err)
			break
		}
		if err := texttesting.CheckText01(clt, []string{texttesting.Text01Name}, i); err != nil {
			t.Error(err)
		}
	}
}
