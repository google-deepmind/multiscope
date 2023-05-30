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

// Package template provides helper functions to execute Go templates.
package template

import (
	"fmt"
	"html/template"
	"io"
	"io/fs"
)

// Execute loads a file from a file system, parse it as a template,
// run the template with the data provided, and write the results
// to the provided writer.
func Execute(w io.Writer, root fs.FS, path string, data any) error {
	file, err := root.Open(path)
	if err != nil {
		return fmt.Errorf("error opening %q: %v", path, err)
	}
	defer file.Close()
	buf, err := io.ReadAll(file)
	if err != nil {
		return fmt.Errorf("cannot read %q: %v", path, err)
	}
	t, err := template.New(path).Parse(string(buf))
	if err != nil {
		return fmt.Errorf("error parsing template %q: %v", path, err)
	}
	if err = t.Execute(w, data); err != nil {
		return fmt.Errorf("error writing content of template %q: %v", path, err)
	}
	return nil
}
