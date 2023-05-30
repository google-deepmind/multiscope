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

// Package httphooks maintains a list of hooks to call when an http server starts.
package httphooks

import "github.com/go-chi/chi/v5"

// Hook is a function called when the http server starts.
type Hook func(*chi.Mux) error

var hooks []Hook

// Register a new hook to call when the http server starts.
func Register(h Hook) {
	hooks = append(hooks, h)
}

// RunAll runs all the hooks.
func RunAll(m *chi.Mux) error {
	for _, hook := range hooks {
		if err := hook(m); err != nil {
			return err
		}
	}
	return nil
}
