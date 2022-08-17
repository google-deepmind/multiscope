package tree

import (
	"multiscope/internal/server/core"
	uisettings "multiscope/wasm/settings"
	"multiscope/wasm/ui"
)

const settingsKey = "tree"

type settings struct {
	settings *uisettings.Settings
	visible  map[core.Key]bool

	errF func(err error)
}

func newSettings(mui ui.UI) *settings {
	s := &settings{
		settings: mui.Settings(),
		visible:  make(map[core.Key]bool),
		errF:     mui.DisplayErr,
	}
	s.settings.Get(settingsKey, &s.visible)
	return s
}

func (s *settings) isVisible(path []string) bool {
	return s.visible[core.ToKey(path)]
}

func (s *settings) hideNode(path []string) {
	delete(s.visible, core.ToKey(path))
	s.settings.Set(settingsKey, s.visible)
}

func (s *settings) showNode(path []string) {
	s.visible[core.ToKey(path)] = true
	s.settings.Set(settingsKey, s.visible)
}
