// Package text implements a HTML text writer for Multiscope.
package text

import (
	"multiscope/internal/mime"
	"multiscope/internal/server/core"
	"multiscope/internal/server/writers/base"
)

// Writer represents a text rubric in the web page.
type Writer struct {
	*base.RawWriter
}

var _ core.Node = (*Writer)(nil)

// NewWriter returns a new writer to display text
func NewWriter() *Writer {
	return &Writer{RawWriter: base.NewRawWriter(mime.PlainText)}
}

// Write writes text as the current data to stream.
func (w *Writer) Write(text string) error {
	_, err := w.RawWriter.Write([]byte(text))
	return err
}
