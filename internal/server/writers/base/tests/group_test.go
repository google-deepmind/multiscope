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
