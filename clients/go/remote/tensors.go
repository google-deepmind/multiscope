package remote

import (
	"context"
	"errors"

	"multiscope/internal/server/writers/tensor"
	pb "multiscope/protos/tensor_go_proto"
	pbgrpc "multiscope/protos/tensor_go_proto"
)

type (
	// Tensor is a minimal interface to describe tensors.
	Tensor = tensor.Tensor

	// TensorWriter writes scalars to Multiscope.
	TensorWriter struct {
		*ClientNode
		clt    pbgrpc.TensorsClient
		writer *pb.Writer
	}
)

// NewTensorWriter creates a new writer to display tensors.
func NewTensorWriter(clt *Client, name string, parent Path) (*TensorWriter, error) {
	clttw := pbgrpc.NewTensorsClient(clt.Connection())
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
	displayPath := rep.DefaultPanelPath
	if displayPath != nil && len(displayPath.Path) > 0 {
		if err := clt.Display().DisplayIfDefault(displayPath.Path); err != nil {
			return nil, err
		}
	}
	writerPath := toPath(writer)
	return &TensorWriter{
		ClientNode: NewClientNode(clt, writerPath),
		clt:        clttw,
		writer:     writer,
	}, nil
}

// Write a tensor.
func (w *TensorWriter) Write(tns tensor.Tensor) error {
	_, err := w.clt.Write(context.Background(), &pb.WriteRequest{
		Writer: w.writer,
		Tensor: tensor.ToProto(tns),
	})
	return err
}
