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
	"fmt"
	"sync"

	"multiscope/internal/mime"
	"multiscope/internal/server/core"
	"multiscope/internal/server/treeservice"
	pb "multiscope/protos/tree_go_proto"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

// ProtoWriter implements a leaf node to stream raw bytes.
type ProtoWriter struct {
	mux  sync.Mutex
	data *anypb.Any
	tick uint32
}

var _ core.Node = (*ProtoWriter)(nil)

func toURL(p proto.Message) string {
	const urlPrefix = "type.googleapis.com/"
	if p == nil {
		return ""
	}
	return urlPrefix + string(proto.MessageName(p))
}

// NewProtoWriter returns a new writer to stream raw bytes.
func NewProtoWriter(p proto.Message) *ProtoWriter {
	w := &ProtoWriter{
		tick: 1,
		data: &anypb.Any{},
	}
	w.data.TypeUrl = toURL(p)
	return w
}

// Write sets the content of the leaf with p.
func (w *ProtoWriter) Write(p proto.Message) error {
	data, err := proto.Marshal(p)
	if err != nil {
		return err
	}
	w.WriteBytes(toURL(p), data)
	return nil
}

// AddToTree adds the proto writer to tree in server state, at the given path.
func (w *ProtoWriter) AddToTree(state *treeservice.State, path *pb.NodePath) (*core.Path, error) {
	return core.SetNodeAt(state.Root(), path, w)
}

func (w *ProtoWriter) mime() string {
	return mime.AnyToMIME(w.data)
}

// MIME returns the MIME type of the last protocol buffer written according to
// go/multiscope-rfc #12.
func (w *ProtoWriter) MIME() string {
	w.mux.Lock()
	defer w.mux.Unlock()
	return w.mime()
}

// WriteBytes sets the content of the leaf with p.
func (w *ProtoWriter) WriteBytes(typeURL string, data []byte) {
	w.mux.Lock()
	defer w.mux.Unlock()
	if typeURL != "" {
		w.data.TypeUrl = typeURL
	}
	w.data.Value = data
	w.tick++
}

// MarshalData writes the raw data into a NodeData protocol buffer.
func (w *ProtoWriter) MarshalData(d *pb.NodeData, path []string, lastTick uint32) {
	w.mux.Lock()
	defer w.mux.Unlock()

	d.Tick = w.tick
	d.Mime = w.mime()
	if lastTick == w.tick {
		return
	}
	d.Data = &pb.NodeData_Pb{Pb: &anypb.Any{
		Value:   w.data.Value, // will be nil if no data has been written yet
		TypeUrl: w.data.TypeUrl,
	}}
}

func (w *ProtoWriter) String() string {
	return w.MIME()
}

// HandleWrite handles a put data request for the proto writer.
func (w *ProtoWriter) HandleWrite(msg proto.Message) error {
	return w.Write(msg)
}

// HandleWriteAny copies the data from a NodeData into this node without dese.
func (w *ProtoWriter) HandleWriteAny(data *pb.NodeData) error {
	if data.GetPb() == nil {
		return fmt.Errorf("`pb` must be set in NodeData for ProtoWriter")
	}
	// Use WriteBytes here as we want to avoid unmarshalling any bytes.
	w.WriteBytes(data.GetPb().GetTypeUrl(), data.GetPb().GetValue())
	return nil
}
