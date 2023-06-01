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

package core_test

import (
	"testing"

	"multiscope/internal/server/core"

	"github.com/google/go-cmp/cmp"
)

func TestKey(t *testing.T) {
	paths := [][]string{
		{"hello", "world"},
		{"hello/world"},
		{"hello/world/"},
		{"hello/world\\/"},
		{"hello/world\\/bonjour"},
		{`hello\`, "world"},
		{`hello\\\\`, "world"},
		{`hel\\\lo\\\\`, "world"},
	}
	for i, path := range paths {
		key := core.ToKey(path)
		got := key.Split()
		if !cmp.Equal(got, path) {
			t.Errorf("key %d error: got %v but want %v", i, got, path)
		}
	}
}
