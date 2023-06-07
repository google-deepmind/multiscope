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
	"reflect"

	pb "multiscope/protos/scalar_go_proto"
	pbgrpc "multiscope/protos/scalar_go_proto"

	"github.com/pkg/errors"
)

// ScalarWriter writes scalars to Multiscope.
type ScalarWriter struct {
	*ClientNode
	clt    pbgrpc.ScalarsClient
	writer *pb.Writer
}

// NewScalarWriter creates a new writer to write scalars to Multiscope.
func NewScalarWriter(clt *Client, name string, parent Path) (*ScalarWriter, error) {
	clw := pbgrpc.NewScalarsClient(clt.Connection())
	ctx := context.Background()
	path := clt.toChildPath(name, parent)
	rep, err := clw.NewWriter(ctx, &pb.NewWriterRequest{
		TreeId: clt.TreeID(),
		Path:   path.NodePath(),
	})
	if err != nil {
		return nil, errors.Errorf("cannot create ScalarWriter: %v", err)
	}
	writer := rep.GetWriter()
	if writer == nil {
		return nil, errors.New("server has returned a nil ScalarWriter")
	}
	writerPath := toPath(writer)
	if err := clt.Display().DisplayIfDefault(writerPath); err != nil {
		return nil, err
	}
	return &ScalarWriter{
		ClientNode: NewClientNode(clt, writerPath),
		clt:        clw,
		writer:     writer,
	}, nil
}

var floatTarget = reflect.TypeOf(float64(0))

func toFloat(v any) (float64, bool) {
	if !reflect.TypeOf(v).ConvertibleTo(floatTarget) {
		return 0, false
	}
	return reflect.ValueOf(v).Convert(floatTarget).Float(), true
}

// Write a set of (key,value) pairs.
// An error is returned if a value cannot be converted to float64.
func (w *ScalarWriter) Write(data map[string]any) error {
	fData := make(map[string]float64)
	for k, v := range data {
		vf, ok := toFloat(v)
		if !ok {
			return errors.Errorf("ScalarWriter error: cannot convert key %q:%T to float64", k, v)
		}
		fData[k] = vf
	}
	return w.WriteFloat64(fData)
}

// WriteFloat64 writes a labeled float64 values.
func (w *ScalarWriter) WriteFloat64(data map[string]float64) error {
	_, err := w.clt.Write(context.Background(), &pb.WriteRequest{
		Writer:       w.writer,
		LabelToValue: data,
	})
	return err
}
