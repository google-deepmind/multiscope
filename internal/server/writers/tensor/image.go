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
