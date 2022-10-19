package tensor

import (
	"multiscope/internal/server/core"
	"multiscope/internal/server/writers/tensor/tnrdr"
)

type bitPlaneUpdater struct {
	parent   *Writer
	img      tnrdr.Image
	writer   *imageWriter
	key      core.Key
	m        *tnrdr.Metrics
	renderer *tnrdr.Renderer
	cache    *bitTensor
}

type bitTensor struct {
	shape []int
	value []float32
}

func resizeBitTensor(t Tensor, old *bitTensor) *bitTensor {
	size := len(t.Values()) * 8
	if old != nil && size == len(old.Values()) {
		return old
	}
	return &bitTensor{value: make([]float32, size)}
}

func updateValues(t Tensor, value []float32) {
	for j, k := range t.Values() {
		for i := 0; i < 8; i++ {
			value[j*8+i] = float32((uint8(k)>>i)&1) * 255
		}
	}
}

func (b *bitTensor) Shape() []int {
	return b.shape
}

func (b *bitTensor) Values() []float32 {
	return b.value
}

func newBitPlaneUpdater(parent *Writer) *bitPlaneUpdater {
	up := &bitPlaneUpdater{
		parent:   parent,
		writer:   newImageWriter(),
		m:        &tnrdr.Metrics{Min: 0, Range: 1},
		renderer: tnrdr.NewRenderer(tnrdr.NewRGBA),
	}
	parent.AddChild(NodeNameBitPlane, up.writer)
	return up
}

func (u *bitPlaneUpdater) forwardActive(parent *core.Path) {
	path := parent.PathTo(NodeNameBitPlane)
	u.parent.state.PathLog().Forward(parent, path)
	u.key = path.ToKey()
}

func (u *bitPlaneUpdater) reset() error {
	u.img = u.renderer.ClearImage(u.img)
	return u.writer.Write(u.img)
}

func (u *bitPlaneUpdater) update(t Tensor) error {
	if !u.parent.state.PathLog().IsActive(u.key) {
		return nil
	}
	u.cache = resizeBitTensor(t, u.cache)
	u.cache.shape = append(append([]int{}, t.Shape()...), 8)
	updateValues(t, u.cache.value)
	if t != nil {
		u.img = u.renderer.Render(u.cache, u.m, u.img)
	}
	return u.writer.Write(u.img)
}
