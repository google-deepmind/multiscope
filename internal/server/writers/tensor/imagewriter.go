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

package tensor

import (
	"image"
	"image/png"
	"multiscope/internal/mime"
	"multiscope/internal/server/writers/base"
)

type imageWriter struct {
	*base.RawWriter
	encoder png.Encoder
}

func newImageWriter() *imageWriter {
	w := &imageWriter{
		RawWriter: base.NewRawWriter(mime.PNG),
	}
	w.encoder.CompressionLevel = png.BestSpeed
	return w
}

func (w *imageWriter) Write(img image.Image) error {
	buf := w.RawWriter.LockBuffer()
	defer w.RawWriter.UnlockBuffer()
	buf.Reset()
	if img.Bounds().Empty() {
		return nil
	}
	return w.encoder.Encode(buf, img)
}
