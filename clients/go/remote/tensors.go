package remote

import (
	"context"
	"errors"
	"fmt"

	"multiscope/lib/tensor"
	pb "multiscope/protos/tensor_go_proto"
	pbgrpc "multiscope/protos/tensor_go_proto"
)

// TensorWriter writes scalars to Multiscope.
type TensorWriter[T tensor.Supported] struct {
	*ClientNode
	clt    pbgrpc.TensorsClient
	writer *pb.Writer
}

// NewTensorWriter creates a new writer to display tensors.
func NewTensorWriter[T tensor.Supported](clt *Client, name string, parent Path) (*TensorWriter[T], error) {
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
	return &TensorWriter[T]{
		ClientNode: NewClientNode(clt, writerPath),
		clt:        clttw,
		writer:     writer,
	}, nil
}

// Write a tensor.
func (w *TensorWriter[T]) Write(tns tensor.Tensor[T]) error {
	tensorPB, err := tensor.Marshal[T](tns)
	if err != nil {
		return fmt.Errorf("cannot marshal tensor: %v", err)
	}
	_, err = w.clt.Write(context.Background(), &pb.WriteRequest{
		Writer: w.writer,
		Tensor: tensorPB,
	})
	return err
}
