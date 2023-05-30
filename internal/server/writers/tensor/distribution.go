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

package tensor

import (
	"math"

	"multiscope/internal/server/core"
	"multiscope/internal/server/writers/plot"
	plotpb "multiscope/protos/plot_go_proto"
)

const (
	numBins        = 20
	xLabel, yLabel = "values", "nums"
)

type distributionUpdater struct {
	indexer
	parent *Writer
	writer *plot.Writer
	bins   []int
	key    core.Key

	plot   *plotpb.Plot
	drawer *plotpb.HistogramDrawer
}

func newDistribution(parent *Writer) *distributionUpdater {
	dist := &distributionUpdater{
		parent: parent,
		writer: plot.NewWriter(),
		bins:   make([]int, numBins),
		drawer: &plotpb.HistogramDrawer{},
	}
	dist.plot = &plotpb.Plot{
		Plotters: []*plotpb.Plotter{
			{
				Drawer: &plotpb.Plotter_HistogramDrawer{
					HistogramDrawer: dist.drawer,
				},
			},
		},
	}
	parent.AddChild(NodeNameDistribution, dist.writer)
	return dist
}

func (u *distributionUpdater) forwardActive(parent *core.Path) {
	path := parent.PathTo(NodeNameDistribution)
	u.parent.state.PathLog().Forward(parent, path)
	u.key = path.ToKey()
}

func (u *distributionUpdater) reset() (err error) {
	u.drawer.Bins = u.drawer.Bins[:0]
	return u.writer.Write(u.plot)
}

func (u *distributionUpdater) update(updateIndex uint, t sTensor) (err error) {
	if !u.parent.state.PathLog().IsActive(u.key) {
		return nil
	}
	return u.forceUpdate(updateIndex, t)
}

func (u *distributionUpdater) forceUpdate(updateIndex uint, _ sTensor) (err error) {
	u.indexer.updateIndex(updateIndex)
	if u.parent.tensor.size() < 2 {
		return u.reset()
	}
	for i := range u.bins {
		u.bins[i] = 0
	}
	bucketSize := ((u.parent.m.Range) / numBins)
	for _, v := range u.parent.tensor.ValuesF32() {
		bucket := int(math.Floor(float64((v - u.parent.m.Min) / bucketSize)))
		if bucket < 0 {
			bucket = 0
		}
		if bucket >= len(u.bins) {
			bucket = len(u.bins) - 1
		}
		u.bins[bucket]++
	}
	if len(u.drawer.Bins) != len(u.bins) {
		u.drawer.Bins = make([]*plotpb.HistogramDrawer_Bin, len(u.bins))
		for i := range u.drawer.Bins {
			u.drawer.Bins[i] = &plotpb.HistogramDrawer_Bin{}
		}
	}
	for i, bin := range u.bins {
		iF32 := float32(i)
		u.drawer.Bins[i].Min = float64(u.parent.m.Min + (bucketSize * iF32))
		u.drawer.Bins[i].Max = float64(u.parent.m.Min + (bucketSize * (iF32 + 1)))
		u.drawer.Bins[i].Weight = float64(bin)
	}
	return u.writer.Write(u.plot)
}
