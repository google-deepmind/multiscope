package base

import (
	"multiscope/internal/server/core"
)

type root struct{ *Group }

// NewRoot returns a new stream root.
func NewRoot() core.Root {
	return &root{Group: NewGroup("")}
}

// Path returns the path to the root node.
func (r root) Path() *core.Path {
	return core.NewPath(r)
}
