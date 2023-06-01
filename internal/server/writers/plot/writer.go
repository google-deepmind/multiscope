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

// Package plot implements a writer for generic plots.
package plot

import (
	"multiscope/internal/server/writers/base"
	plotpb "multiscope/protos/plot_go_proto"
)

// Writer writes histogram data for the frontend.
type Writer struct {
	*base.ProtoWriter
}

// NewWriter returns a writer to write a histogram.
func NewWriter() *Writer {
	w := &Writer{}
	w.ProtoWriter = base.NewProtoWriter(&plotpb.Plot{})
	return w
}

// Write the latest histogram data.
func (w *Writer) Write(data *plotpb.Plot) error {
	return w.ProtoWriter.Write(data)
}
