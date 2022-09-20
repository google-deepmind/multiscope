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
}

const settingLeftBarVisible = "leftbar_visible"

func newLeftBar(ui *UI) (*LeftBar, error) {
	const leftClass = "container__left"
	elements := ui.Owner().GetElementsByClassName(leftClass)
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
	if l.isVisible() {
		l.show()
	} else {
		l.hide()
	}
	return l, nil
}

func (l *LeftBar) isVisible() bool {
	var visible bool
	l.ui.Settings().Get(settingLeftBarVisible, &visible)
	return visible
}

func (l *LeftBar) show() {
	l.ui.Settings().Set(settingLeftBarVisible, true)
	l.root.Style().SetProperty("display", "block", "")
}

func (l *LeftBar) hide() {
	l.ui.Settings().Set(settingLeftBarVisible, false)
	l.root.Style().SetProperty("display", "none", "")
}
