package ui

import (
	"github.com/pkg/errors"
	"honnef.co/go/js/dom/v2"
)

type LeftBar struct {
	ui   *UI
	root dom.HTMLElement
}

const (
	settingLeftBar = "leftbar_visible"
	visibleSetting = "visible"
)

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
	if l.isVisible() {
		l.show()
	} else {
		l.hide()
	}
	return l, nil
}

func (l *LeftBar) isVisible() bool {
	v, _ := l.ui.Settings().Get(settingLeftBar)
	return v == visibleSetting
}

func (l *LeftBar) show() {
	l.ui.Settings().Set(settingLeftBar, visibleSetting)
	l.root.Style().SetProperty("display", "block", "")
}

func (l *LeftBar) hide() {
	l.ui.Settings().Set(settingLeftBar, "")
	l.root.Style().SetProperty("display", "none", "")
}
