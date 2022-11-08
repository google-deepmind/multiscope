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
	ui.Owner().Doc().AddEventListener("keypress", false, func(ev dom.Event) {
		kev := ev.(*dom.KeyboardEvent)
		char := string([]byte{byte(kev.CharCode())})
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
