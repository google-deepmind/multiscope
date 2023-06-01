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

	pb "multiscope/protos/text_go_proto"
	pbgrpc "multiscope/protos/text_go_proto"

	"github.com/pkg/errors"
)

// HTMLWriter writes raw text to Multiscope.
type HTMLWriter struct {
	*ClientNode
	clt    pbgrpc.TextClient
	writer *pb.HTMLWriter
}

// NewHTMLWriter creates a new writer to write html and css to Multiscope.
func NewHTMLWriter(clt *Client, name string, parent Path) (*HTMLWriter, error) {
	clttw := pbgrpc.NewTextClient(clt.Connection())
	ctx := context.Background()
	path := clt.toChildPath(name, parent)
	rep, err := clttw.NewHTMLWriter(ctx, &pb.NewHTMLWriterRequest{
		TreeId: clt.TreeID(),
		Path:   path.NodePath(),
	})
	if err != nil {
		return nil, errors.Errorf("cannot create HTMLWriter: %v", err)
	}
	writer := rep.GetWriter()
	if writer == nil {
		return nil, errors.Errorf("server has returned a nil HTMLWriter")
	}
	writerPath := toPath(writer)
	if err := clt.Display().DisplayIfDefault(writerPath); err != nil {
		return nil, err
	}
	return &HTMLWriter{
		ClientNode: NewClientNode(clt, writerPath),
		clt:        clttw,
		writer:     writer,
	}, nil
}

// Write html to the Multiscope display.
func (w *HTMLWriter) Write(html string) error {
	_, err := w.clt.WriteHTML(context.Background(), &pb.WriteHTMLRequest{
		Writer: w.writer,
		Html:   html,
	})
	if err != nil {
		return errors.Errorf("cannot write HTML  data: %v", err)
	}
	return err
}

// WriteCSS css to the Multiscope display.
func (w *HTMLWriter) WriteCSS(css string) error {
	_, err := w.clt.WriteCSS(context.Background(), &pb.WriteCSSRequest{
		Writer: w.writer,
		Css:    css,
	})
	if err != nil {
		return errors.Errorf("cannot write CSS data: %v", err)
	}
	return err
}
