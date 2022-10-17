package remote

import (
	"context"
	"errors"

	"multiscope/internal/server/writers/tensor"
	pb "multiscope/protos/tensor_go_proto"
	pbgrpc "multiscope/protos/tensor_go_proto"
)

// TensorWriter writes scalars to Multiscope.
type TensorWriter struct {
	*ClientNode
	clt    pbgrpc.TensorsClient
	writer *pb.Writer
}

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
	writerPath := toPath(writer)
	if err := clt.Display().DisplayIfDefault(writerPath); err != nil {
		return nil, err
	}
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
