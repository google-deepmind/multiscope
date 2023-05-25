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

package reflect_test

import (
	"context"
	"fmt"
	"testing"

	"multiscope/clients/go/clienttesting"
	"multiscope/clients/go/reflect"
	"multiscope/clients/go/remote"
	"multiscope/internal/grpc/client"

	"github.com/google/go-cmp/cmp"
)

const (
	customNodeName = "bonus"
	customNodeData = "Youpi: a TextWriter in addition of all the other writers"
)

type customParser struct{}

func (customParser) CanParse(obj any) bool {
	_, ok := obj.(*toParseType)
	return ok
}

func (p *customParser) Parse(state *reflect.ParserState, name string, fObj reflect.TargetGetter) (remote.Node, error) {
	grp, err := state.Parse(name, fObj, p)
	if err != nil {
		return nil, err
	}
	clt := state.Root().Client()
	writer, err := remote.NewTextWriter(clt, customNodeName, grp.Path())
	if err != nil {
		return nil, err
	}
	writer.Write(customNodeData)
	return grp, nil
}

func TestParser(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	reflect.RegisterParser(&customParser{})
	ticker, toParse, err := createToParseTestingNodes(clt)
	if err != nil {
		t.Fatal(err)
	}
	// Force all the nodes to be active.
	if err := clienttesting.ForceActive(clt.TreeClient(), ticker.Path()); err != nil {
		t.Fatal(err)
	}
	// Tick to write the data.
	if err := ticker.Tick(); err != nil {
		t.Fatal(err)
	}
	// Check the data that has been written.
	dataGot, err := queryTreeUntilNotEmpty(clt, ticker, toNodePaths(reflectDataWant))
	if err != nil {
		t.Error(err)
	}
	if !cmp.Equal(dataGot, reflectDataWant) {
		t.Errorf("wrong values: got %v want %v", dataGot, reflectDataWant)
	}

	// Check that we have an additional node with some data in it.
	nodeName := fmt.Sprintf("%s:%T", toParseName, toParse)
	const ctName = "ct:*reflect_test.childType"
	paths := [][]string{
		ticker.Path().Append(nodeName, customNodeName),
	}
	ctx := context.Background()
	nodes, err := client.PathToNodes(ctx, clt.TreeClient(), paths...)
	if err != nil {
		t.Fatal(err)
	}
	if nodes[0].GetError() != "" {
		t.Fatalf("bonus node error: %s", nodes[0].GetError())
	}
	data, err := client.NodesData(ctx, clt.TreeClient(), nodes)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := client.ToRaw(data[0])
	if err != nil {
		t.Fatal(err)
	}
	got := string(raw)
	if got != customNodeData {
		t.Errorf("bonus node does not have the correct data: got %q but want %q", got, customNodeData)
	}
}
