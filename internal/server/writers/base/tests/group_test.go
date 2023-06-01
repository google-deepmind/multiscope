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

package tree_test

import (
	"testing"

	"multiscope/internal/server/writers/base"
)

func TestGroupAddChild(t *testing.T) {
	grp := base.NewGroup("")
	tests := []struct {
		base      string
		wantNames []string
		wantNodes []*base.Group
	}{
		{
			base:      "child",
			wantNames: []string{"child", "child1", "child2"},
			wantNodes: []*base.Group{base.NewGroup(""), base.NewGroup(""), base.NewGroup("")},
		},
		{
			base:      "anotherChild",
			wantNames: []string{"anotherChild", "anotherChild1"},
			wantNodes: []*base.Group{base.NewGroup(""), base.NewGroup("")},
		},
	}
	for _, test := range tests {
		for i, child := range test.wantNodes {
			got := grp.AddChild(test.base, child)
			if got != test.wantNames[i] {
				t.Errorf("child name error: got %q but want %q", got, test.wantNames[i])
			}
		}
	}
	for _, test := range tests {
		for i, name := range test.wantNames {
			got, err := grp.Child(name)
			if err != nil {
				t.Error(err)
			}
			if got != test.wantNodes[i] {
				t.Errorf("incorrect child returned by the group: got %v want %v", got, test.wantNodes[i])
			}
		}
	}
}
