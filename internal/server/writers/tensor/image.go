package tensor

import (
	"multiscope/internal/server/core"
	"multiscope/internal/server/writers/tensor/tnrdr"
)

type imageUpdater struct {
	indexer
	parent   *Writer
	img      tnrdr.Image
	writer   *imageWriter
	renderer *tnrdr.Renderer
	key      core.Key
}

func newImageUpdater(parent *Writer) *imageUpdater {
	up := &imageUpdater{
		parent:   parent,
		writer:   newImageWriter(),
		renderer: tnrdr.NewRenderer(tnrdr.NewRGBA),
	}
	parent.AddChild(NodeNameImage, up.writer)
	return up
}

func (u *imageUpdater) forwardActive(parent *core.Path) {
	path := parent.PathTo(NodeNameImage)
	u.parent.state.PathLog().Forward(parent, path)
	u.key = path.ToKey()
}

func (u *imageUpdater) reset() error {
	u.img = u.renderer.ClearImage(u.img)
	return u.writer.Write(u.img)
}

func (u *imageUpdater) update(updateIndex uint, t sTensor) error {
	if !u.parent.state.PathLog().IsActive(u.key) {
		return nil
	}
	return u.forceUpdate(updateIndex, t)
}

func (u *imageUpdater) forceUpdate(updateIndex uint, t sTensor) error {
	u.updateIndex(updateIndex)
	u.img = u.renderer.Render(t, &u.parent.m, u.img)
	return u.writer.Write(u.img)
}
