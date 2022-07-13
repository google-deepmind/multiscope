package remote

import (
	"context"
	"errors"

	pb "multiscope/protos/text_go_proto"
	pbgrpc "multiscope/protos/text_go_proto"
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
		Path: path.NodePath(),
	})
	if err != nil {
		return nil, err
	}
	writer := rep.GetWriter()
	if writer == nil {
		return nil, errors.New("server has returned a nil TensorWriter")
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
	return err
}

// WriteCSS css to the Multiscope display.
func (w *HTMLWriter) WriteCSS(css string) error {
	_, err := w.clt.WriteCSS(context.Background(), &pb.WriteCSSRequest{
		Writer: w.writer,
		Css:    css,
	})
	return err
}
