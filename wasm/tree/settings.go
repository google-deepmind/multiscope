package tree

import (
	"multiscope/internal/server/core"
	uisettings "multiscope/wasm/settings"
)

const settingsKey = "tree"

type settings struct {
	el       *Element
	settings *uisettings.Settings
	visible  map[core.Key]bool

	errF func(err error)
}

func newSettings(el *Element) *settings {
	return &settings{
		el:       el,
		settings: el.ui.Settings(),
		visible:  make(map[core.Key]bool),
		errF:     el.ui.DisplayErr,
	}
}

func (s *settings) registerListener() {
	s.settings.Listen(settingsKey, &s.visible, s.el.refresh)
}

func (s *settings) isVisible(path []string) bool {
	return s.visible[core.ToKey(path)]
}

func (s *settings) hideNode(path []string) {
	delete(s.visible, core.ToKey(path))
	s.settings.Set(s, settingsKey, s.visible)
}

func (s *settings) showNode(path []string) {
	s.visible[core.ToKey(path)] = true
	s.settings.Set(s, settingsKey, s.visible)
}
