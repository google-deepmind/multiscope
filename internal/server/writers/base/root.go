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
