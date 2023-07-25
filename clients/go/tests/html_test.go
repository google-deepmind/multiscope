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

func TestHTMLWriter(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	writer, err := remote.NewHTMLWriter(clt, texttesting.HTML01Name, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.WriteCSS(texttesting.CSS01Data); err != nil {
		t.Fatal(err)
	}
	for i, want := range texttesting.HTML01Data {
		if err := writer.Write(want); err != nil {
			t.Error(err)
			break
		}
		if err := texttesting.CheckHTML01(clt, []string{texttesting.Text01Name}, i); err != nil {
			t.Error(err)
		}
	}
}

func TestHTMLWriterIO(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	writer, err := remote.NewHTMLWriter(clt, texttesting.HTML01Name, nil)
	if err != nil {
		t.Fatal(err)
	}
	tmpl := template.Must(template.New("").Parse("{{.content}}"))
	if err := tmpl.Execute(writer.CSSIO(), map[string]string{
		"content": texttesting.CSS01Data,
	}); err != nil {
		t.Error(err)
	}
	for i, want := range texttesting.HTML01Data {
		if err := tmpl.Execute(writer.HTMLIO(), map[string]string{
			"content": want,
		}); err != nil {
			t.Error(err)
			break
		}
		if err := texttesting.CheckHTML01(clt, []string{texttesting.Text01Name}, i); err != nil {
			t.Error(err)
		}
	}
}
