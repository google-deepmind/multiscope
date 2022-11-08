//go:build js

// Package injector provides dependency injection on the frontend.
package injector

import (
	"multiscope/internal/style"
	"multiscope/wasm/ui"
)

// UI using the injected code.
type UI interface {
	Style() *style.Style

	Owner() *ui.Owner
}

var deps = []func(UI){}

// Inject a new dependency into the application.
func Inject(f func(UI)) {
	deps = append(deps, f)
}

// Run all the dependencies.
func Run(ui UI) {
	for _, f := range deps {
		f(ui)
	}
}
