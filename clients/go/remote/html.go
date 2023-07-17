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
	"fmt"
	"io"

	pb "multiscope/protos/text_go_proto"
	pbgrpc "multiscope/protos/text_go_proto"
	treepb "multiscope/protos/tree_go_proto"

	"github.com/pkg/errors"
)

type (
	htmlIO struct {
		w *HTMLWriter
	}

	cssIO struct {
		w *HTMLWriter
	}

	// HTMLWriter writes raw text to Multiscope.
	HTMLWriter struct {
		*ClientNode
		clt    pbgrpc.TextClient
		writer *pb.HTMLWriter

		callback Callback
	}
)

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
	node, err := NewClientNode(clt, writerPath)
	if err != nil {
		return nil, err
	}
	w := &HTMLWriter{
		ClientNode: node,
		clt:        clttw,
		writer:     writer,
	}
	clt.EventsManager().NewQueueForPath(w.Path(), w.processEvent)
	return w, nil
}

func (w *HTMLWriter) processEvent(ev *treepb.Event) error {
	if w.callback == nil {
		return nil
	}
	return w.callback(ev)
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

// WriteStringer writes html to the Multiscope display.
// If the argument implements the Listener interface, then all events
// coming to the writer are forward to the listener.
// If a listener was registered before, it is unregistered.
func (w *HTMLWriter) WriteStringer(html fmt.Stringer) error {
	if err := w.Write(html.String()); err != nil {
		return err
	}
	callbacker, ok := html.(Callbacker)
	if ok {
		w.callback = callbacker.Callback()
	} else {
		w.callback = nil
	}
	return nil
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

// HTMLIO adapts a HTMLWriter into a io.Writer to write the HTML.
func (w *HTMLWriter) HTMLIO() io.Writer {
	return htmlIO{w: w}
}

func (wio htmlIO) Write(data []byte) (int, error) {
	return len(data), wio.w.Write(string(data))
}

// CSSIO adapts a HTMLWriter into a io.Writer to write the CSS.
func (w *HTMLWriter) CSSIO() io.Writer {
	return cssIO{w: w}
}

func (wio cssIO) Write(data []byte) (int, error) {
	return len(data), wio.w.WriteCSS(string(data))
}
