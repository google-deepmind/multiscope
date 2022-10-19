package tensor

import (
	"fmt"
	"math"

	"multiscope/internal/server/core"
	"multiscope/internal/server/writers/text"

	"github.com/russross/blackfriday/v2"
)

type tensorUpdater struct {
	parent *Writer
	writer *text.HTMLWriter
}

func newTensorUpdater(parent *Writer) *tensorUpdater {
	u := &tensorUpdater{
		parent: parent,
		writer: text.NewHTMLWriter(),
	}
	parent.AddChild(NodeNameInfo, u.writer)
	return u
}

func (u *tensorUpdater) forwardActive(parent *core.Path) {
	path := parent.PathTo(NodeNameInfo)
	u.writer.SetForwards(u.parent.state, path)
	u.parent.state.PathLog().Forward(parent, path)
}

const tensorInfo = `
* **Shape**: %v
* **Size**: %d
* **Minimum value**: %f
* **Maximum value**: %f
`

func (u *tensorUpdater) reset() error {
	return u.update(nil)
}

func (u *tensorUpdater) update(Tensor) error {
	size := len(u.parent.tensor.Values())
	min := u.parent.m.HistMin
	max := u.parent.m.HistMax
	if size == 0 {
		min = float32(math.NaN())
		max = float32(math.NaN())
	}
	t := fmt.Sprintf(tensorInfo, u.parent.tensor.Shape(), size, min, max)
	if size > 0 && size < 100 {
		t += fmt.Sprintf("\n\n**String representation**:\n```\n%s\n```", toString(u.parent.tensor))
	}
	return u.writer.Write(string(blackfriday.Run([]byte(t))))
}
