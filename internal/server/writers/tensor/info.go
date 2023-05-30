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
	"math"

	"multiscope/internal/server/core"
	"multiscope/internal/server/writers/text"

	"github.com/russross/blackfriday/v2"
)

type tensorUpdater struct {
	indexer
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
	return u.update(0, nil)
}

func (u *tensorUpdater) update(updateIndex uint, t sTensor) error {
	return u.forceUpdate(updateIndex, t)
}

func (u *tensorUpdater) forceUpdate(updateIndex uint, _ sTensor) error {
	u.indexer.updateIndex(updateIndex)

	size := u.parent.tensor.size()
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
