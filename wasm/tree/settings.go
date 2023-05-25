// Copyright 2023 Google LLC
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
