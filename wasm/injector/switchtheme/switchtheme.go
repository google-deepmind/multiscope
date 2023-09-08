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

//go:build js

// Package switchtheme provides a shortcut to change the color theme.
package switchtheme

import (
	"sort"

	"multiscope/wasm/injector"

	"github.com/tdegris/base16-go/themes"
	"honnef.co/go/js/dom/v2"
)

var names []string

func init() {
	injector.Inject(switchtheme)
	for k := range themes.Base16 {
		names = append(names, k)
	}
	sort.Strings(names)
}

func switchtheme(ui injector.UI) {
	current := 0
	theme := ui.Style().Theme().Name
	for i, name := range names {
		if name == theme {
			current = i
			break
		}
	}
	ui.Owner().Doc().AddEventListener("keydown", false, func(ev dom.Event) {
		kev := ev.(*dom.KeyboardEvent)
		char := string([]byte{byte(kev.KeyCode())})
		if char == "t" {
			current = (current + 1) % len(names)
			ui.Style().SetTheme(names[current])
		}
		if char == "T" {
			current--
			if current < 0 {
				current = len(names) - 1
			}
			ui.Style().SetTheme(names[current])
		}
	})
}
