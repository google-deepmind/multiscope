package storage_test

import (
	"multiscope/internal/server/writers/ticker/storage"
	"testing"
)

func TestStorage(t *testing.T) {
	local := &storage.Storage{}
	local.SetSize(100)
	for i := 0; i < 10; i++ {
		local.Register()
		defer local.Unregister()
	}
	maxWant := uint64(100 / 10)
	maxGot := local.Available()
	if maxGot != maxWant {
		t.Errorf("maximum storage is incorrect: got %d but want %d", maxGot, maxWant)
	}
}
