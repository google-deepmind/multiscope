package tensor

import (
	"math"

	"multiscope/internal/server/core"
	"multiscope/internal/server/writers/histogram"
	tablepb "multiscope/protos/table_go_proto"
)

const numBins = 20

type distributionUpdater struct {
	parent *Writer
	writer *histogram.Writer
	tbl    *tablepb.Series
	bins   []int
	key    core.Key
}

func newDistribution(parent *Writer) *distributionUpdater {
	dist := &distributionUpdater{
		parent: parent,
		writer: histogram.NewWriter(),
		bins:   make([]int, numBins),
		tbl:    &tablepb.Series{},
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
	u.tbl.Reset()
	return u.writer.Write(u.tbl)
}

func (u *distributionUpdater) update(Tensor) (err error) {
	if !u.parent.state.PathLog().IsActive(u.key) {
		return nil
	}
	if len(u.parent.tensor.Values()) < 2 {
		u.tbl.Reset()
		return u.writer.Write(u.tbl)
	}
	for i := range u.bins {
		u.bins[i] = 0
	}
	bucketSize := ((u.parent.m.Range) / numBins)
	for _, v := range u.parent.tensor.Values() {
		bucket := int(math.Floor(float64((v - u.parent.m.Min) / bucketSize)))
		if bucket < 0 {
			bucket = 0
		}
		if bucket >= len(u.bins) {
			bucket = len(u.bins) - 1
		}
		u.bins[bucket]++
	}
	u.tbl.Reset()
	serie := tablepb.Serie{}
	x := u.parent.m.Min + bucketSize/2
	for _, bin := range u.bins {
		serie.Points = append(serie.Points, &tablepb.Point{
			X: float64(x),
			Y: float64(bin),
		})
		x += bucketSize
	}
	u.tbl.LabelToSerie = map[string]*tablepb.Serie{
		"values": &serie,
	}
	return u.writer.Write(u.tbl)
}
