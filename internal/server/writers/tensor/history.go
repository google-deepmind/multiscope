package tensor

import (
	"fmt"
	"multiscope/internal/server/core"
	"multiscope/internal/server/writers/scalar"
)

type historyUpdater struct {
	indexer
	parent           *Writer
	w                *scalar.Writer
	nSelect, lastLen int
	data             map[string]float64
}

func newHistory(parent *Writer) *historyUpdater {
	w := scalar.NewWriter()
	parent.AddChild(NodeNameHistory, w)
	return &historyUpdater{
		parent:  parent,
		w:       w,
		lastLen: -1,
	}
}

func (u *historyUpdater) forwardActive(parent *core.Path) {
	historyPath := parent.PathTo(NodeNameHistory)
	u.parent.state.PathLog().Forward(parent, historyPath)
}

func (u *historyUpdater) reset() error {
	u.w.Reset()
	return nil
}

func (u *historyUpdater) update(updateIndex uint, t Tensor) error {
	return u.forceUpdate(updateIndex, t)
}

func (u *historyUpdater) forceUpdate(updateIndex uint, _ Tensor) error {
	u.indexer.updateIndex(updateIndex)
	const maxIndicesSelected = 20

	vals := u.parent.tensor.Values()
	if len(vals) != u.lastLen {
		u.data = make(map[string]float64)
		u.nSelect = maxIndicesSelected
		if len(vals) < u.nSelect {
			u.nSelect = len(vals)
		}
		u.lastLen = len(vals)
	}

	for i, v := range vals[:u.nSelect] {
		u.data[fmt.Sprintf("%01d", i)] = float64(v)
	}
	return u.w.Write(u.data)
}
