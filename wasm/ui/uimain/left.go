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

package uimain

import (
	"multiscope/wasm/tree"

	"github.com/pkg/errors"
	"honnef.co/go/js/dom/v2"
)

// LeftBar implements the tree and its container on the left.
type LeftBar struct {
	ui   *UI
	root dom.HTMLElement
	tree *tree.Element

	visibleSetting bool
}

const settingLeftBarVisible = "leftbar_visible"

func newLeftBar(ui *UI) (*LeftBar, error) {
	const leftClass = "container_left"
	elements := ui.Owner().Doc().GetElementsByClassName(leftClass)
	if len(elements) != 1 {
		return nil, errors.Errorf("wrong number of elements of class %q: got %d but want 1", leftClass, len(elements))
	}
	l := &LeftBar{
		ui:   ui,
		root: elements[0].(dom.HTMLElement),
	}
	var err error
	if l.tree, err = tree.NewElement(ui); err != nil {
		return l, err
	}
	l.root.AppendChild(l.tree.Root())
	l.ui.Settings().Listen(settingLeftBarVisible, &l.visibleSetting, func(any) error {
		if l.visibleSetting {
			l.root.Style().SetProperty("display", "block", "")
		} else {
			l.root.Style().SetProperty("display", "none", "")
		}
		return nil
	})
	return l, nil
}

func (l *LeftBar) isVisible() bool {
	return l.visibleSetting
}

func (l *LeftBar) show() {
	l.ui.Settings().Set(l, settingLeftBarVisible, true)
}

func (l *LeftBar) hide() {
	l.ui.Settings().Set(l, settingLeftBarVisible, false)
}
