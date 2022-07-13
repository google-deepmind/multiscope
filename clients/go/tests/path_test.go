package scope_test

import (
	"testing"

	"multiscope/clients/go/remote"
)

func TestHasPrefix(t *testing.T) {
	pre := remote.Path([]string{"a", "b"})
	pth := remote.Path([]string{"a", "b", "c"})
	if !pth.HasPrefix(pre) {
		t.Errorf("%v.HasPrefix(%v) should return true", pth, pre)
	}
	pth = remote.Path([]string{"a1", "b", "c"})
	if pth.HasPrefix(pre) {
		t.Errorf("%v.HasPrefix(%v) should return false", pth, pre)
	}
	pth = remote.Path([]string{"a"})
	if pth.HasPrefix(pre) {
		t.Errorf("%v.HasPrefix(%v) should return false", pth, pre)
	}
}
