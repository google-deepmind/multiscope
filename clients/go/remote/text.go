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
		return nil, errors.New("server has returned a nil TensorWriter")
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
