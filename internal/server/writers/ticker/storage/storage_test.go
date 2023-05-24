// Copyright 2023 Google LLC
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
