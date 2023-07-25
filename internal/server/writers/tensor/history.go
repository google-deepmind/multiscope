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
	u.w.ResetNode()
	return nil
}

func (u *historyUpdater) update(updateIndex uint, t sTensor) error {
	return u.forceUpdate(updateIndex, t)
}

func (u *historyUpdater) forceUpdate(updateIndex uint, _ sTensor) error {
	u.indexer.updateIndex(updateIndex)
	const maxIndicesSelected = 20

	vals := u.parent.tensor.ValuesF32()
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
