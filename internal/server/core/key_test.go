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
