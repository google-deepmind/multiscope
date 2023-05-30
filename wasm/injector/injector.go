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
