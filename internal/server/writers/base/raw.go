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

package base

import (
	"bytes"
	"sync"

	"multiscope/internal/server/core"
	"multiscope/internal/server/treeservice"
	pb "multiscope/protos/tree_go_proto"
)

// RawWriter implements a leaf node to stream raw bytes.
type RawWriter struct {
	mux  sync.Mutex
	mime string
	data bytes.Buffer
	tick uint32
}

var _ core.Node = (*RawWriter)(nil)

// NewRawWriter returns a new writer to stream raw bytes.
func NewRawWriter(mime string) *RawWriter {
	return &RawWriter{
		tick: 1,
		mime: mime,
	}
}

// ResetNode resets the node.
func (w *RawWriter) ResetNode() error {
	buf := w.LockBuffer()
	defer w.UnlockBuffer()

	buf.Reset()
	return nil
}

// Write sets the content of the leaf with p.
func (w *RawWriter) Write(p []byte) error {
	buf := w.LockBuffer()
	defer w.UnlockBuffer()

	buf.Reset()
	_, err := buf.Write(p)
	return err
}

// AddToTree adds the raw writer to tree in server state, at the given path.
func (w *RawWriter) AddToTree(state *treeservice.State, path *pb.NodePath) (*core.Path, error) {
	return core.SetNodeAt(state.Root(), path, w)
}

// MIME returns the MIME type of the raw data.
func (w *RawWriter) MIME() string {
	return w.mime
}

// MarshalData writes the raw data into a NodeData protocol buffer.
func (w *RawWriter) MarshalData(d *pb.NodeData, path []string, lastTick uint32) {
	w.mux.Lock()
	defer w.mux.Unlock()

	d.Tick = w.tick
	d.Mime = w.mime
	if w.tick == lastTick {
		return
	}

	p := &pb.NodeData_Raw{}
	p.Raw = append([]byte{}, w.data.Bytes()...)
	d.Data = p
}

// LockBuffer locks the writer and returns the table to manipulate its data.
func (w *RawWriter) LockBuffer() *bytes.Buffer {
	w.mux.Lock()
	return &w.data
}

// UnlockBuffer unlocks the writer. The table returned by LockTable() should not be modified after this point.
func (w *RawWriter) UnlockBuffer() {
	w.tick++
	w.mux.Unlock()
}

func (w *RawWriter) String() string {
	return w.MIME()
}
