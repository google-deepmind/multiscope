// Package histogram implements a writer to display histograms.
package histogram

import (
	"multiscope/internal/server/writers/base"
	tablepb "multiscope/protos/table_go_proto"
)

// Writer writes histogram data for the frontend.
type Writer struct {
	*base.ProtoWriter
}

// NewWriter returns a writer to write a histogram.
func NewWriter() *Writer {
	w := &Writer{}
	w.ProtoWriter = base.NewProtoWriter(&tablepb.Series{})
	return w
}

// Write the latest histogram data.
func (w *Writer) Write(data *tablepb.Series) error {
	return w.ProtoWriter.Write(data)
}
