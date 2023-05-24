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

package scope_test

import (
	"multiscope/clients/go/clienttesting"
	"multiscope/clients/go/remote"
	"multiscope/internal/server/writers/ticker/tickertesting"
	"testing"
)

func TestTicker(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	ticker, err := remote.NewTicker(clt, tickertesting.Ticker01Name, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := clienttesting.ForceActive(clt.TreeClient(), ticker.Path()); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 42; i++ {
		if err := ticker.Tick(); err != nil {
			t.Fatal(err)
		}
	}
	if err := tickertesting.CheckRemoteTicker(clt.TreeClient(), ticker.Path().NodePath().GetPath(), ticker.CurrentTick()); err != nil {
		t.Errorf("wrong data on the server: %v", err)
	}
}
