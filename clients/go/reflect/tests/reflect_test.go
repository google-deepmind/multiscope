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
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"multiscope/clients/go/clienttesting"
	"multiscope/clients/go/reflect"
	"multiscope/clients/go/remote"
	"multiscope/internal/grpc/client"
	plotpb "multiscope/protos/plot_go_proto"
	streampb "multiscope/protos/tree_go_proto"

	"github.com/google/go-cmp/cmp"
	"go.uber.org/multierr"
)

type (
	ChildInterface interface {
		A() float32
	}

	ChildType struct {
		childA float32
		childB float32
	}

	toParseType struct {
		a   float32
		b   float64
		c   int
		d   uint
		ct  *ChildType `multiscope_reflect:"Child"`
		Exp *ChildType
	}
)

func (ct *ChildType) A() float32 {
	return ct.childA
}

func (tt *toParseType) Child() *ChildType {
	return tt.ct
}

func extractValues(paths [][]string, data []*streampb.NodeData) (map[string]float32, error) {
	var err error
	res := make(map[string]float32)
	for nodeI, node := range data {
		tbl := &plotpb.ScalarsPlot{}
		nodeErr := client.ToProto(node, tbl)
		if nodeErr != nil {
			nodeErr = fmt.Errorf("path %v error: %v", paths[nodeI], nodeErr)
			err = multierr.Append(err, nodeErr)
			continue
		}
		if tbl.Plot == nil || len(tbl.Plot.Plotters) == 0 {
			continue
		}
		for _, plot := range tbl.Plot.Plotters {
			drawer := plot.GetLineDrawer()
			for _, point := range drawer.GetPoints() {
				path := node.GetPath().GetPath()
				label := strings.Join(path, "/") + "[" + plot.Legend + "]"
				res[label] = float32(point.Y)
			}
		}
	}
	return res, err
}

const toParseName = "toParse"

func createToParseTestingNodes(clt *remote.Client) (ticker *remote.Ticker, toParse *toParseType, err error) {
	ticker, err = remote.NewTicker(clt, "Ticker", nil)
	if err != nil {
		return
	}
	toParse = &toParseType{
		a:   42,
		b:   43,
		c:   44,
		d:   45,
		ct:  &ChildType{childA: 142, childB: 144},
		Exp: &ChildType{childA: 242, childB: 244},
	}
	if _, err = reflect.On(ticker, toParseName, toParse); err != nil {
		return nil, nil, err
	}
	return
}

func checkTreeData(clt *remote.Client, ticker *remote.Ticker, paths [][]string) (map[string]float32, error) {
	const ctName = "ct:*reflect_test.ChildType"
	const expName = "Exp:*reflect_test.ChildType"
	ctx := context.Background()
	nodes, err := client.PathToNodes(ctx, clt, paths...)
	if err != nil {
		return nil, err
	}
	for i, node := range nodes {
		if node.GetError() != "" {
			return nil, fmt.Errorf("%v error: %v", paths[i], node.GetError())
		}
	}
	data, err := client.NodesData(ctx, clt, nodes)
	if err != nil {
		return nil, err
	}
	got, err := extractValues(paths, data)
	if err != nil {
		return nil, err
	}
	return got, nil
}

func queryTreeUntilNotEmpty(clt *remote.Client, ticker *remote.Ticker, paths [][]string) (got map[string]float32, err error) {
	start := time.Now()
	for len(got) != len(paths) && err == nil {
		if err := ticker.Tick(); err != nil {
			break
		}
		if got, err = checkTreeData(clt, ticker, paths); err != nil {
			break
		}
		if time.Since(start) > 10*time.Second {
			err = errors.New("timeout: could not get data from the server")
		}
	}
	return
}

var reflectDataWant = map[string]float32{
	"Ticker/toParse:*reflect_test.toParseType/a[a]":                                       42,
	"Ticker/toParse:*reflect_test.toParseType/b[b]":                                       43,
	"Ticker/toParse:*reflect_test.toParseType/c[c]":                                       44,
	"Ticker/toParse:*reflect_test.toParseType/d[d]":                                       45,
	"Ticker/toParse:*reflect_test.toParseType/ct:*reflect_test.ChildType/childA[childA]":  142,
	"Ticker/toParse:*reflect_test.toParseType/ct:*reflect_test.ChildType/childB[childB]":  144,
	"Ticker/toParse:*reflect_test.toParseType/Exp:*reflect_test.ChildType/childA[childA]": 242,
	"Ticker/toParse:*reflect_test.toParseType/Exp:*reflect_test.ChildType/childB[childB]": 244,
}

func toNodePaths(want map[string]float32) [][]string {
	paths := [][]string{}
	for path := range want {
		path = path[0:strings.LastIndex(path, "[")]
		paths = append(paths, strings.Split(path, "/"))
	}
	return paths
}

func TestReflect(t *testing.T) {
	clt, err := clienttesting.Start()
	if err != nil {
		t.Fatal(err)
	}
	ticker, _, err := createToParseTestingNodes(clt)
	if err != nil {
		t.Fatal(err)
	}
	// Tick to write the data.
	if err := ticker.Tick(); err != nil {
		t.Fatal(err)
	}
	// Check that no data has been written.
	paths := toNodePaths(reflectDataWant)
	_, err = checkTreeData(clt, ticker, paths)
	if err != nil {
		t.Error(err)
	}
	// Write and query data until the result is not empty.
	got, err := queryTreeUntilNotEmpty(clt, ticker, paths)
	if err != nil {
		t.Fatal(err)
	}
	if !cmp.Equal(got, reflectDataWant) {
		t.Errorf("wrong values: got %v want %v", got, reflectDataWant)
	}
}
