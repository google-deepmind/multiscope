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
