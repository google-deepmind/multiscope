// Copyright 2023 Google LLC
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
	"bytes"
	"context"
	"image"
	"image/png"
	"multiscope/internal/mime"

	pb "multiscope/protos/base_go_proto"
	pbgrpc "multiscope/protos/base_go_proto"

	"github.com/pkg/errors"
)

// ImageWriter writes images to Multiscope.
type ImageWriter struct {
	*ClientNode
	clt    pbgrpc.BaseWritersClient
	writer *pb.RawWriter

	buffer  *bytes.Buffer
	encoder png.Encoder
}

// NewImageWriter returns a new writer to display images in Multiscope.
func NewImageWriter(clt *Client, name string, parent Path) (*ImageWriter, error) {
	clw := pbgrpc.NewBaseWritersClient(clt.Connection())
	path := clt.toChildPath(name, parent)
	rep, err := clw.NewRawWriter(context.Background(), &pb.NewRawWriterRequest{
		Path: path.NodePath(),
		Mime: mime.PNG,
	})
	if err != nil {
		return nil, err
	}
	writer := rep.GetWriter()
	if writer == nil {
		return nil, errors.New("server has returned a nil ImageWriter")
	}
	writerPath := toPath(writer)
	if err := clt.Display().DisplayIfDefault(writerPath); err != nil {
		return nil, err
	}
	w := &ImageWriter{
		ClientNode: NewClientNode(clt, path),
		clt:        clw,
		buffer:     &bytes.Buffer{},
		writer:     writer,
	}
	w.encoder.CompressionLevel = png.BestSpeed
	return w, nil
}

// Write an image to the Multiscope display.
func (w *ImageWriter) Write(img image.Image) error {
	w.buffer.Reset()
	if err := w.encoder.Encode(w.buffer, img); err != nil {
		return err
	}
	_, err := w.clt.WriteRaw(context.Background(), &pb.WriteRawRequest{
		Writer: w.writer,
		Data:   w.buffer.Bytes(),
	})
	return err
}
