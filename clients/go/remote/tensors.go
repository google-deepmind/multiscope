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
	"fmt"

	"multiscope/lib/tensor"
	pb "multiscope/protos/tensor_go_proto"
	pbgrpc "multiscope/protos/tensor_go_proto"
)

type (
	// TensorWriterPB writes tensors already represented as protocol buffers.
	TensorWriterPB struct {
		*ClientNode
		clt    pbgrpc.TensorsClient
		writer *pb.Writer
	}

	// TensorWriter writes tensors to Multiscope.
	TensorWriter[T tensor.Supported] struct {
		*TensorWriterPB
	}
)

// NewTensorWriterPB creates a new writer to display tensors already represented as protocol buffers.
func NewTensorWriterPB(clt *Client, name string, parent Path) (*TensorWriterPB, error) {
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
	return &TensorWriterPB{
		ClientNode: NewClientNode(clt, writerPath),
		clt:        clttw,
		writer:     writer,
	}, nil
}

// WritePB a tensor as a protocol buffer.
func (w *TensorWriterPB) WritePB(tPB *pb.Tensor) error {
	_, err := w.clt.Write(context.Background(), &pb.WriteRequest{
		Writer: w.writer,
		Tensor: tPB,
	})
	return err
}

// NewTensorWriter creates a new writer to display tensors.
func NewTensorWriter[T tensor.Supported](clt *Client, name string, parent Path) (*TensorWriter[T], error) {
	w, err := NewTensorWriterPB(clt, name, parent)
	if err != nil {
		return nil, err
	}
	return &TensorWriter[T]{TensorWriterPB: w}, nil
}

// Write a tensor.
func (w *TensorWriter[T]) Write(tns tensor.Tensor[T]) error {
	tensorPB, err := tensor.Marshal[T](tns)
	if err != nil {
		return fmt.Errorf("cannot marshal tensor: %v", err)
	}
	return w.WritePB(tensorPB)
}
