// Package plot implements a writer for generic plots.
package plot

import (
	"multiscope/internal/server/writers/base"
	plotpb "multiscope/protos/plot_go_proto"
)

// Writer writes histogram data for the frontend.
type Writer struct {
	*base.ProtoWriter
}

// NewWriter returns a writer to write a histogram.
func NewWriter() *Writer {
	w := &Writer{}
	w.ProtoWriter = base.NewProtoWriter(&plotpb.Plot{})
	return w
}

// Write the latest histogram data.
func (w *Writer) Write(data *plotpb.Plot) error {
	return w.ProtoWriter.Write(data)
}
