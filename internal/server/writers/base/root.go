package base

import (
	"fmt"
	"multiscope/internal/server/core"

	"github.com/pkg/errors"
)

type (
	deleter interface {
		DeleteChild(name string) error
	}

	root struct{ *Group }
)

// NewRoot returns a new stream root.
func NewRoot() core.Root {
	return &root{Group: NewGroup("")}
}

// Path returns the path to the root node.
func (r *root) Path() *core.Path {
	return core.NewPath(r)
}

// Delete deletes a node in the tree.
func (r *root) Delete(path []string) error {
	if len(path) == 0 {
		return fmt.Errorf("cannot delete the root node: not implemented")
	}
	parent, err := core.PathToNode(r, path[:len(path)-1])
	if err != nil {
		return err
	}
	dltr, ok := parent.(deleter)
	if !ok {
		return errors.Errorf("cannot cast type %T to base.deleter", parent)
	}
	return dltr.DeleteChild(path[len(path)-1])
}
