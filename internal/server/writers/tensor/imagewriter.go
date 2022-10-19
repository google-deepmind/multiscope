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
