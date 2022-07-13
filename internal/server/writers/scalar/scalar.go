// Package scalar implements a scalar time series writer for Multiscope.
package scalar

import (
	"sync"

	"multiscope/internal/server/core"
	"multiscope/internal/server/treeservice"
	"multiscope/internal/server/writers/base"
	tablepb "multiscope/protos/table_go_proto"
	treepb "multiscope/protos/tree_go_proto"
)

// Writer writes a time series of scalars.
type Writer struct {
	*base.ProtoWriter
	mut             sync.Mutex
	defaultTimeStep int
	data            *tablepb.Series
}

var _ core.Node = (*Writer)(nil)

// TimeLabel is a label to use to override the default time counter of the vega writer.
const TimeLabel = "__time__"

var historyLength = 200

// NewWriter returns a vega writer collecting data to be plotted.
func NewWriter() *Writer {
	w := &Writer{}
	w.ProtoWriter = base.NewProtoWriter(w.data)
	w.Reset()
	return w
}

// AddToTree adds the writer to a stream tree.
// Also set up activity forwarding.
func (w *Writer) AddToTree(state treeservice.State, path *treepb.NodePath) (*core.Path, error) {
	return core.SetNodeAt(state.Root(), path, w)
}

// removeHead makes we only remember the last w.historyLength elements
func (w *Writer) removeHeads() {
	for _, serie := range w.data.LabelToSerie {
		if len(serie.Points) <= historyLength {
			continue
		}
		serie.Points = serie.Points[len(serie.Points)-historyLength:]
	}
}

// Reset resets the writer data.
func (w *Writer) Reset() {
	w.mut.Lock()
	defer w.mut.Unlock()
	w.data = &tablepb.Series{
		LabelToSerie: make(map[string]*tablepb.Serie),
	}
}

// Write accumulates new float64 to plot.
func (w *Writer) Write(d map[string]float64) error {
	w.mut.Lock()
	defer w.mut.Unlock()
	time, ok := d[TimeLabel]
	if !ok {
		time = float64(w.defaultTimeStep)
		defer func() { w.defaultTimeStep++ }()
	}
	for key, value := range d {
		if key == TimeLabel {
			continue
		}
		serie := w.data.LabelToSerie[key]
		if serie == nil {
			serie = &tablepb.Serie{}
			w.data.LabelToSerie[key] = serie
		}
		serie.Points = append(serie.Points, &tablepb.Point{X: time, Y: value})
	}
	w.removeHeads()
	return w.ProtoWriter.Write(w.data)
}

// SetHistoryLength sets how much historical data to keep around.
// (Use only for tests.)
func SetHistoryLength(hl int) {
	historyLength = hl
}
