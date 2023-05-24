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

// Package text implements a HTML text writer for Multiscope.
package text

import (
	"multiscope/internal/mime"
	"multiscope/internal/server/core"
	"multiscope/internal/server/writers/base"
)

// Writer represents a text rubric in the web page.
type Writer struct {
	*base.RawWriter
}

var _ core.Node = (*Writer)(nil)

// NewWriter returns a new writer to display text.
func NewWriter() *Writer {
	return &Writer{RawWriter: base.NewRawWriter(mime.PlainText)}
}

// Write writes text as the current data to stream.
func (w *Writer) Write(text string) error {
	return w.RawWriter.Write([]byte(text))
}
