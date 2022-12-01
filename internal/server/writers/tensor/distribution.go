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
	parent *Writer
	writer *plot.Writer
	bins   []int
	key    core.Key

	serie *plotpb.Serie
	plot  *plotpb.Plot
}

func newDistribution(parent *Writer) *distributionUpdater {
	dist := &distributionUpdater{
		parent: parent,
		writer: plot.NewWriter(),
		bins:   make([]int, numBins),
		serie:  &plotpb.Serie{},
	}
	dist.plot = &plotpb.Plot{
		Plotters: []*plotpb.Plotter{
			{
				Serie: dist.serie,
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
	u.serie.Points = u.serie.Points[:0]
	return u.writer.Write(u.plot)
}

func (u *distributionUpdater) update(Tensor) (err error) {
	if !u.parent.state.PathLog().IsActive(u.key) {
		return nil
	}
	if len(u.parent.tensor.Values()) < 2 {
		return u.reset()
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
	u.serie.Points = u.serie.Points[:0]
	x := u.parent.m.Min + bucketSize/2
	for _, bin := range u.bins {
		u.serie.Points = append(u.serie.Points, &plotpb.Point{
			X: float64(x),
			Y: float64(bin),
		})
		x += bucketSize
	}
	return u.writer.Write(u.plot)
}
