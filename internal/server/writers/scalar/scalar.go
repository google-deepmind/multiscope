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

// Package scalar implements a scalar time series writer for Multiscope.
package scalar

import (
	"sort"
	"sync"

	"multiscope/internal/server/core"
	"multiscope/internal/server/treeservice"
	"multiscope/internal/server/writers/base"
	plotpb "multiscope/protos/plot_go_proto"
	treepb "multiscope/protos/tree_go_proto"

	"golang.org/x/exp/maps"
)

// Writer writes a time series of scalars.
type Writer struct {
	*base.ProtoWriter
	mut             sync.Mutex
	defaultTimeStep int
	plot            *plotpb.ScalarsPlot
	plotters        map[string]*plotpb.Plotter
}

var (
	_ core.Node        = (*Writer)(nil)
	_ core.NodeReseter = (*Writer)(nil)
)

// TimeLabel is a label to use to override the default time counter of the vega writer.
const TimeLabel = "__time__"

var historyLength = 200

// NewWriter returns a writer collecting data to be plotted.
func NewWriter() *Writer {
	w := &Writer{}
	w.reset()
	w.ProtoWriter = base.NewProtoWriter(w.plot)
	return w
}

func (w *Writer) reset() {
	w.mut.Lock()
	defer w.mut.Unlock()
	w.plot = &plotpb.ScalarsPlot{Plot: &plotpb.Plot{}}
	w.plotters = make(map[string]*plotpb.Plotter)
	w.defaultTimeStep = 0
}

// ResetNode resets the writer data.
func (w *Writer) ResetNode() error {
	w.reset()
	return nil
}

// AddToTree adds the writer to a stream tree.
// Also set up activity forwarding.
func (w *Writer) AddToTree(state *treeservice.State, path *treepb.NodePath) (*core.Path, error) {
	return core.SetNodeAt(state.Root(), path, w)
}

func drawer(plotter *plotpb.Plotter) *plotpb.LineDrawer {
	return plotter.Drawer.(*plotpb.Plotter_LineDrawer).LineDrawer
}

// removeHead makes we only remember the last w.historyLength elements.
func (w *Writer) removeHeads() {
	for _, pltr := range w.plotters {
		drawer := drawer(pltr)
		if len(drawer.Points) <= historyLength {
			continue
		}
		drawer.Points = drawer.Points[len(drawer.Points)-historyLength:]
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
	keys := maps.Keys(d)
	sort.Strings(keys)
	for _, key := range keys {
		if key == TimeLabel {
			continue
		}
		value := d[key]
		pltr := w.plotters[key]
		if pltr == nil {
			pltr = w.newPlotter(key)
		}
		drawer := drawer(pltr)
		drawer.Points = append(drawer.Points, &plotpb.LineDrawer_Point{X: time, Y: value})
	}
	w.removeHeads()
	return w.ProtoWriter.Write(w.plot)
}

func (w *Writer) newPlotter(key string) *plotpb.Plotter {
	pltr := &plotpb.Plotter{
		Drawer: &plotpb.Plotter_LineDrawer{
			LineDrawer: &plotpb.LineDrawer{},
		},
		Legend: key,
	}
	w.plotters[key] = pltr
	w.plot.Plot.Plotters = append(w.plot.Plot.Plotters, pltr)
	return pltr
}

// SetHistoryLength sets how much historical data to keep around.
// (Use only for tests.)
func SetHistoryLength(hl int) {
	historyLength = hl
}
