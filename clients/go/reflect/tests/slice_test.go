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

package reflect_test

import (
	"testing"

	"multiscope/clients/go/clienttesting"
	"multiscope/clients/go/reflect"
	"multiscope/clients/go/remote"

	"github.com/google/go-cmp/cmp"
)

type withSlice struct {
	slice1 []*ChildType `multiscope_reflect:"Slice1"`
	Slice2 []*ChildType
	Slice3 []ChildInterface
}

func (w *withSlice) Slice1() []*ChildType {
	return w.slice1
}

func createToParseWithSlice(clt *remote.Client) (ticker *remote.Ticker, toParse *withSlice, err error) {
	ticker, err = remote.NewTicker(clt, "Ticker", nil)
	if err != nil {
		return
	}
	toParse = &withSlice{
		slice1: []*ChildType{
			{childA: 100, childB: 110},
			{childA: 101, childB: 111},
			{childA: 102, childB: 112},
		},
		Slice2: []*ChildType{
			{childA: 200, childB: 210},
			{childA: 201, childB: 211},
			{childA: 202, childB: 212},
		},
		Slice3: []ChildInterface{
			&ChildType{childA: 300, childB: 310},
			&ChildType{childA: 301, childB: 311},
			&ChildType{childA: 302, childB: 312},
		},
	}
	if _, err = reflect.On(ticker, toParseName, toParse); err != nil {
		return nil, nil, err
	}
	return
}

var withSliceDataWant = map[string]float32{
	"Ticker/toParse:*reflect_test.withSlice/slice1:[]*reflect_test.ChildType/0:*reflect_test.ChildType/childA[childA]": 100,
	"Ticker/toParse:*reflect_test.withSlice/slice1:[]*reflect_test.ChildType/0:*reflect_test.ChildType/childB[childB]": 110,
	"Ticker/toParse:*reflect_test.withSlice/slice1:[]*reflect_test.ChildType/1:*reflect_test.ChildType/childA[childA]": 101,
	"Ticker/toParse:*reflect_test.withSlice/slice1:[]*reflect_test.ChildType/1:*reflect_test.ChildType/childB[childB]": 111,
	"Ticker/toParse:*reflect_test.withSlice/slice1:[]*reflect_test.ChildType/2:*reflect_test.ChildType/childA[childA]": 102,
	"Ticker/toParse:*reflect_test.withSlice/slice1:[]*reflect_test.ChildType/2:*reflect_test.ChildType/childB[childB]": 112,

	"Ticker/toParse:*reflect_test.withSlice/Slice2:[]*reflect_test.ChildType/0:*reflect_test.ChildType/childA[childA]": 200,
	"Ticker/toParse:*reflect_test.withSlice/Slice2:[]*reflect_test.ChildType/0:*reflect_test.ChildType/childB[childB]": 210,
	"Ticker/toParse:*reflect_test.withSlice/Slice2:[]*reflect_test.ChildType/1:*reflect_test.ChildType/childA[childA]": 201,
	"Ticker/toParse:*reflect_test.withSlice/Slice2:[]*reflect_test.ChildType/1:*reflect_test.ChildType/childB[childB]": 211,
	"Ticker/toParse:*reflect_test.withSlice/Slice2:[]*reflect_test.ChildType/2:*reflect_test.ChildType/childA[childA]": 202,
	"Ticker/toParse:*reflect_test.withSlice/Slice2:[]*reflect_test.ChildType/2:*reflect_test.ChildType/childB[childB]": 212,

	"Ticker/toParse:*reflect_test.withSlice/Slice3:[]reflect_test.ChildInterface/0:*reflect_test.ChildType/childA[childA]": 300,
	"Ticker/toParse:*reflect_test.withSlice/Slice3:[]reflect_test.ChildInterface/0:*reflect_test.ChildType/childB[childB]": 310,
	"Ticker/toParse:*reflect_test.withSlice/Slice3:[]reflect_test.ChildInterface/1:*reflect_test.ChildType/childA[childA]": 301,
	"Ticker/toParse:*reflect_test.withSlice/Slice3:[]reflect_test.ChildInterface/1:*reflect_test.ChildType/childB[childB]": 311,
	"Ticker/toParse:*reflect_test.withSlice/Slice3:[]reflect_test.ChildInterface/2:*reflect_test.ChildType/childA[childA]": 302,
	"Ticker/toParse:*reflect_test.withSlice/Slice3:[]reflect_test.ChildInterface/2:*reflect_test.ChildType/childB[childB]": 312,
}

func TestReflectFieldArray(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	ticker, _, err := createToParseWithSlice(clt)
	if err != nil {
		t.Fatal(err)
	}
	// Tick to write the data.
	if err := ticker.Tick(); err != nil {
		t.Fatal(err)
	}
	// Write and query data until the result is not empty.
	paths := toNodePaths(withSliceDataWant)
	got, err := queryTreeUntilNotEmpty(clt, ticker, paths)
	if err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(got, withSliceDataWant); diff != "" {
		t.Errorf("wrong values: got %v want %v\ndiff:\n%s", got, withSliceDataWant, diff)
	}
}
