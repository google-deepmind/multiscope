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

package remote

import (
	"context"
	"errors"

	pb "multiscope/protos/text_go_proto"
	pbgrpc "multiscope/protos/text_go_proto"
)

// TextWriter writes raw text to Multiscope.
type TextWriter struct {
	*ClientNode
	clt    pbgrpc.TextClient
	writer *pb.Writer
}

// NewTextWriter creates a new writer to write text to Multiscope.
func NewTextWriter(clt *Client, name string, parent Path) (*TextWriter, error) {
	clttw := pbgrpc.NewTextClient(clt.Connection())
	ctx := context.Background()
	path := clt.toChildPath(name, parent)
	rep, err := clttw.NewWriter(ctx, &pb.NewWriterRequest{
		Path: path.NodePath(),
	})
	if err != nil {
		return nil, err
	}
	writer := rep.GetWriter()
	if writer == nil {
		return nil, errors.New("server has returned a nil TextWriter")
	}
	writerPath := toPath(writer)
	if err := clt.Display().DisplayIfDefault(writerPath); err != nil {
		return nil, err
	}
	return &TextWriter{
		ClientNode: NewClientNode(clt, writerPath),
		clt:        clttw,
		writer:     writer,
	}, nil
}

// Write raw text to the Multiscope display.
func (w *TextWriter) Write(text string) error {
	_, err := w.clt.Write(context.Background(), &pb.WriteRequest{
		Writer: w.writer,
		Text:   text,
	})
	return err
}
