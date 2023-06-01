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

// Package testing exports function to test the Tensor writer.
package testing

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"math"

	"multiscope/internal/grpc/client"
	"multiscope/internal/mime"
	scopetensor "multiscope/internal/server/writers/tensor"
	scopetesting "multiscope/internal/testing"
	"multiscope/lib/tensor"
	plotpb "multiscope/protos/plot_go_proto"
	treepb "multiscope/protos/tree_go_proto"

	"github.com/google/go-cmp/cmp"
	"go.uber.org/multierr"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/testing/protocmp"
)

type (
	tensorF32 interface {
		tensor.Base
		Values() []float32
		ValuesF32() []float32
	}

	tensorImpl struct {
		shape []int
		value []float32
	}
)

// Shape returns the shape of the testing tensor.
func (t *tensorImpl) Shape() []int {
	return t.shape
}

// Values returns the set of values of the testing tensor.
func (t *tensorImpl) Values() []float32 {
	return t.value
}

// ValuesF32 returns the set of values of the testing tensor.
func (t *tensorImpl) ValuesF32() []float32 {
	return t.value
}

// Tensor01Name is the name of the tensor writer in the tree.
const Tensor01Name = "tensor01"

type testData struct {
	Desc               string
	Tensor             tensorF32
	ToImg, ToBitPlane  func() *image.RGBA
	Min, Max           float32
	ScaleMin, ScaleMax float32
	L1Norm, L2Norm     float32
	Dist               *plotpb.Plot
	Info               string
}

func distributionToSerie(dist [][]float64) []*plotpb.HistogramDrawer_Bin {
	bins := []*plotpb.HistogramDrawer_Bin{}
	if len(dist) == 0 {
		return bins
	}
	for i := range dist {
		bins = append(bins, &plotpb.HistogramDrawer_Bin{
			Min:    dist[i][0],
			Max:    dist[i][1],
			Weight: dist[i][2],
		})
	}
	return bins
}

func buildDistribution(dist [][]float64) *plotpb.Plot {
	return &plotpb.Plot{
		Plotters: []*plotpb.Plotter{{
			Drawer: &plotpb.Plotter_HistogramDrawer{
				HistogramDrawer: &plotpb.HistogramDrawer{
					Bins: distributionToSerie(dist),
				},
			},
		}},
	}
}

func tolerance(epsilon float64) cmp.Option {
	return cmp.Comparer(func(x, y float64) bool {
		delta := math.Abs(x - y)
		mean := math.Abs(x+y) / 2.0
		return delta/mean < epsilon
	})
}

var (
	resetData = testData{
		Desc: "after reset",
		Tensor: &tensorImpl{
			shape: nil,
			value: []float32{},
		},
		ToImg: func() *image.RGBA {
			return nil
		},
		ToBitPlane: func() *image.RGBA {
			return nil
		},
		Dist: buildDistribution(nil),
		Info: `<ul>
<li><strong>Shape</strong>: []</li>
<li><strong>Size</strong>: 0</li>
<li><strong>Minimum value</strong>: NaN</li>
<li><strong>Maximum value</strong>: NaN</li>
</ul>
`,
	}

	// TensorTests is a list of check to perform when testing a Tensor writer.
	TensorTests = []testData{
		{
			Desc: "empty tensor",
			Tensor: &tensorImpl{
				shape: nil,
				value: []float32{},
			},
			ToImg: func() *image.RGBA {
				return nil
			},
			ToBitPlane: func() *image.RGBA {
				return nil
			},
			Dist:     buildDistribution(nil),
			Min:      math.MaxFloat32,
			Max:      -math.MaxFloat32,
			ScaleMin: math.MaxFloat32,
			ScaleMax: -math.MaxFloat32,
			Info: `<ul>
<li><strong>Shape</strong>: []</li>
<li><strong>Size</strong>: 0</li>
<li><strong>Minimum value</strong>: NaN</li>
<li><strong>Maximum value</strong>: NaN</li>
</ul>
`,
		},
		{
			Desc: "scalar tensor",
			Tensor: &tensorImpl{
				shape: []int{1},
				value: []float32{200},
			},
			ToImg: func() *image.RGBA {
				img := image.NewRGBA(image.Rect(0, 0, 1, 1))
				img.Pix = []byte{0x0, 0x0, 0x0, 0xff}
				return img
			},
			ToBitPlane: func() *image.RGBA {
				img := image.NewRGBA(image.Rect(0, 0, 3, 3))
				img.Pix = []byte{0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0x0}
				return img
			},
			Min:      200,
			Max:      200,
			L1Norm:   200,
			L2Norm:   200,
			ScaleMin: 200,
			ScaleMax: 200,
			Dist:     buildDistribution(nil),
			Info: `<ul>
<li><strong>Shape</strong>: [1]</li>
<li><strong>Size</strong>: 1</li>
<li><strong>Minimum value</strong>: 200.000000</li>
<li><strong>Maximum value</strong>: 200.000000</li>
</ul>

<p><strong>String representation</strong>:</p>

<pre><code>[200]
</code></pre>
`,
		},
		{
			Desc: "positive tensor",
			Tensor: &tensorImpl{
				shape: []int{1, 2, 3},
				value: []float32{1, 2, 3, 4, 5, 6},
			},
			ToImg: func() *image.RGBA {
				img := image.NewRGBA(image.Rect(0, 0, 2, 3))
				img.Pix = []byte{0x3f, 0x1e, 0x13, 0xff, 0xb2, 0x55, 0x35, 0xff, 0x66, 0x30, 0x1e, 0xff, 0xd8, 0x67, 0x41, 0xff, 0x8c, 0x43, 0x2a, 0xff, 0xff, 0x7a, 0x4d, 0xff}
				return img
			},
			ToBitPlane: func() *image.RGBA {
				img := image.NewRGBA(image.Rect(0, 0, 16, 3))
				img.Pix = []byte{0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff}
				return img
			},
			Min:      1,
			Max:      6,
			L1Norm:   1 + 2 + 3 + 4 + 5 + 6,
			L2Norm:   float32(math.Sqrt(1 + 4 + 9 + 16 + 25 + 36)),
			ScaleMin: 1,
			ScaleMax: 200,
			Dist: buildDistribution([][]float64{
				{1.000000, 1.250000, 1.000000},
				{1.250000, 1.500000, 0.000000},
				{1.500000, 1.750000, 0.000000},
				{1.750000, 2.000000, 0.000000},
				{2.000000, 2.250000, 1.000000},
				{2.250000, 2.500000, 0.000000},
				{2.500000, 2.750000, 0.000000},
				{2.750000, 3.000000, 0.000000},
				{3.000000, 3.250000, 1.000000},
				{3.250000, 3.500000, 0.000000},
				{3.500000, 3.750000, 0.000000},
				{3.750000, 4.000000, 0.000000},
				{4.000000, 4.250000, 1.000000},
				{4.250000, 4.500000, 0.000000},
				{4.500000, 4.750000, 0.000000},
				{4.750000, 5.000000, 0.000000},
				{5.000000, 5.250000, 1.000000},
				{5.250000, 5.500000, 0.000000},
				{5.500000, 5.750000, 0.000000},
				{5.750000, 6.000000, 1.000000},
			}),
			Info: `<ul>
<li><strong>Shape</strong>: [1 2 3]</li>
<li><strong>Size</strong>: 6</li>
<li><strong>Minimum value</strong>: 1.000000</li>
<li><strong>Maximum value</strong>: 200.000000</li>
</ul>

<p><strong>String representation</strong>:</p>

<pre><code>[0,.,.]=
1,2,3,
4,5,6,

</code></pre>
`,
		},
		{
			Desc: "negative tensor",
			Tensor: &tensorImpl{
				shape: []int{1, 2, 3},
				value: []float32{-6, -5, -4, -3, -2, -1},
			},
			ToImg: func() *image.RGBA {
				img := image.NewRGBA(image.Rect(0, 0, 2, 3))
				img.Pix = []byte{0x6b, 0xfa, 0xff, 0xff, 0x3a, 0x89, 0x8c, 0xff, 0x5a, 0xd4, 0xd8, 0xff, 0x2a, 0x64, 0x66, 0xff, 0x4a, 0xaf, 0xb2, 0xff, 0x1a, 0x3e, 0x3f, 0xff}
				return img
			},
			ToBitPlane: func() *image.RGBA {
				img := image.NewRGBA(image.Rect(0, 0, 16, 3))
				img.Pix = []byte{0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff}
				return img
			},
			Min:      -6,
			Max:      -1,
			L1Norm:   1 + 2 + 3 + 4 + 5 + 6,
			L2Norm:   float32(math.Sqrt(1 + 4 + 9 + 16 + 25 + 36)),
			ScaleMin: -6,
			ScaleMax: 200,
			Dist: buildDistribution([][]float64{
				{-6.000000, -5.750000, 1.000000},
				{-5.750000, -5.500000, 0.000000},
				{-5.500000, -5.250000, 0.000000},
				{-5.250000, -5.000000, 0.000000},
				{-5.000000, -4.750000, 1.000000},
				{-4.750000, -4.500000, 0.000000},
				{-4.500000, -4.250000, 0.000000},
				{-4.250000, -4.000000, 0.000000},
				{-4.000000, -3.750000, 1.000000},
				{-3.750000, -3.500000, 0.000000},
				{-3.500000, -3.250000, 0.000000},
				{-3.250000, -3.000000, 0.000000},
				{-3.000000, -2.750000, 1.000000},
				{-2.750000, -2.500000, 0.000000},
				{-2.500000, -2.250000, 0.000000},
				{-2.250000, -2.000000, 0.000000},
				{-2.000000, -1.750000, 1.000000},
				{-1.750000, -1.500000, 0.000000},
				{-1.500000, -1.250000, 0.000000},
				{-1.250000, -1.000000, 1.000000},
			}),
			Info: `<ul>
<li><strong>Shape</strong>: [1 2 3]</li>
<li><strong>Size</strong>: 6</li>
<li><strong>Minimum value</strong>: -6.000000</li>
<li><strong>Maximum value</strong>: 200.000000</li>
</ul>

<p><strong>String representation</strong>:</p>

<pre><code>[0,.,.]=
-6,-5,-4,
-3,-2,-1,

</code></pre>
`,
		},
		{
			Desc: "negative and positive tensor",
			Tensor: &tensorImpl{
				shape: []int{1, 2, 3},
				value: []float32{-6, 5, 0, 3, -2, 1},
			},
			ToImg: func() *image.RGBA {
				img := image.NewRGBA(image.Rect(0, 0, 2, 3))
				img.Pix = []byte{0x6b, 0xfa, 0xff, 0xff, 0x9f, 0x4c, 0x30, 0xff, 0xdf, 0x6a, 0x43, 0xff, 0x35, 0x7d, 0x7f, 0xff, 0x0, 0x0, 0x0, 0xff, 0x5f, 0x2d, 0x1c, 0xff}
				return img
			},
			ToBitPlane: func() *image.RGBA {
				img := image.NewRGBA(image.Rect(0, 0, 16, 3))
				img.Pix = []byte{0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0xc0, 0x43, 0x99, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff, 0x0, 0x0, 0x0, 0xff}
				return img
			},
			Min:      -6,
			Max:      5,
			L1Norm:   1 + 2 + 3 + 0 + 5 + 6,
			L2Norm:   float32(math.Sqrt(1 + 4 + 9 + 0 + 25 + 36)),
			ScaleMin: -6,
			ScaleMax: 200,
			Dist: buildDistribution([][]float64{
				{-6, -5.44999980926513671875, 1},
				{-5.44999980926513671875, -4.90000009536743164062, 0},
				{-4.90000009536743164062, -4.34999990463256835938, 0},
				{-4.34999990463256835938, -3.79999995231628417969, 0},
				{-3.79999995231628417969, -3.25, 0},
				{-3.25, -2.69999980926513671875, 0},
				{-2.69999980926513671875, -2.14999985694885253906, 0},
				{-2.14999985694885253906, -1.59999990463256835938, 1},
				{-1.59999990463256835938, -1.04999971389770507812, 0},
				{-1.04999971389770507812, -0.5, 0},
				{-0.5, 0.05000019073486328125, 1},
				{0.05000019073486328125, 0.60000038146972656250, 0},
				{0.60000038146972656250, 1.15000009536743164062, 1},
				{1.15000009536743164062, 1.70000028610229492188, 0},
				{1.70000028610229492188, 2.25, 0},
				{2.25, 2.80000019073486328125, 0},
				{2.80000019073486328125, 3.35000038146972656250, 1},
				{3.35000038146972656250, 3.90000057220458984375, 0},
				{3.90000057220458984375, 4.44999980926513671875, 0},
				{4.44999980926513671875, 5., 1},
			}),
			Info: `<ul>
<li><strong>Shape</strong>: [1 2 3]</li>
<li><strong>Size</strong>: 6</li>
<li><strong>Minimum value</strong>: -6.000000</li>
<li><strong>Maximum value</strong>: 200.000000</li>
</ul>

<p><strong>String representation</strong>:</p>

<pre><code>[0,.,.]=
-6,5,0,
3,-2,1,

</code></pre>
`,
		},
	}
)

// NewTestTensor returns a tensor for tests.
func NewTestTensor() tensor.Base {
	return &tensorImpl{
		shape: []int{1, 2, 3},
		value: []float32{-6, 5, -4, 3, -2, 1},
	}
}

// checkRenderedImage checks the data that can be read from the tensor01 node.
func checkRenderedImage(clt client.Client, path []string, nodeName string, tns tensorF32, want *image.RGBA) error {
	ctx := context.Background()
	imgPath := append(append([]string{}, path...), nodeName)
	// Get the nodes and check their types.
	nodes, err := client.PathToNodes(ctx, clt, imgPath)
	if err != nil {
		return err
	}
	if err := scopetesting.CheckNodePaths(nodes); err != nil {
		return err
	}
	if err := scopetesting.CheckMIME(nodes[0].GetMime(), mime.PNG); err != nil {
		return err
	}
	// Check the data returned by the server.
	data, err := client.NodesData(ctx, clt, nodes)
	if err != nil {
		return err
	}
	imgB, err := client.ToRaw(data[0])
	if len(tns.ValuesF32()) == 0 {
		if len(imgB) > 0 {
			return fmt.Errorf("got raw data %v for an empty tensor but want an empty buffer", imgB)
		}
		return nil
	}
	if err != nil {
		return err
	}
	src, err := png.Decode(bytes.NewReader(imgB))
	if err != nil {
		return err
	}

	b := src.Bounds()
	got := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(got, got.Bounds(), src, b.Min, draw.Src)
	if diff := cmp.Diff(want, got); len(diff) > 0 {
		return fmt.Errorf("unexpected image data: %s \n got: %#v \n want: %#v", diff, got.Pix, want.Pix)
	}
	return nil
}

func extractScalarPlotData(data *treepb.NodeData) (*plotpb.Plot, map[string]float32, error) {
	plt := plotpb.ScalarsPlot{}
	if err := client.ToProto(data, &plt); err != nil {
		return nil, nil, err
	}
	vals := make(map[string]float32)
	for _, pltr := range plt.Plot.Plotters {
		drawer := pltr.Drawer.(*plotpb.Plotter_LineDrawer).LineDrawer
		vals[pltr.Legend] = float32(drawer.Points[len(drawer.Points)-1].Y)
	}
	return plt.Plot, vals, nil
}

func checkMetrics(clt client.Client, path []string, test *testData) error {
	ctx := context.Background()
	paths := [][]string{
		append(append([]string{}, path...), scopetensor.NodeNameMinMax),
		append(append([]string{}, path...), scopetensor.NodeNameNorms),
	}
	nodes, err := client.PathToNodes(ctx, clt, paths...)
	if err != nil {
		return err
	}
	if err := scopetesting.CheckNodePaths(nodes); err != nil {
		return err
	}
	var wantMIME = mime.ProtoToMIME(&plotpb.ScalarsPlot{})
	if err := scopetesting.CheckMIME(nodes[0].GetMime(), wantMIME); err != nil {
		return fmt.Errorf("incorrect MIME type for node %v: %v", paths[0], err)
	}
	if err := scopetesting.CheckMIME(nodes[1].GetMime(), wantMIME); err != nil {
		return fmt.Errorf("incorrect MIME type for node %v: %v", paths[1], err)
	}
	// Fetch data returned by the server.
	data, err := client.NodesData(ctx, clt, nodes)
	if err != nil {
		return fmt.Errorf("error while fetching the data: %s", err)
	}
	tbl, vals, dErr := extractScalarPlotData(data[0])
	// Check that the table is empty after the writer has been reset.
	if test == &resetData {
		if len(tbl.Plotters) > 0 {
			return fmt.Errorf("data table should be empty but it is not:\n%v", tbl)
		}
		return nil
	}
	// No reset: check that the data we got is correct.
	err = multierr.Append(err, dErr)
	if dErr == nil && vals["min"] != test.Min {
		err = multierr.Append(err, fmt.Errorf("wrong min value: got %f, want %f. Table:\n%v", vals["min"], test.Min, tbl))
	}
	if dErr == nil && vals["max"] != test.Max {
		err = multierr.Append(err, fmt.Errorf("wrong max value: got %f, want %f. Table:\n%v", vals["max"], test.Max, tbl))
	}
	tbl, vals, dErr = extractScalarPlotData(data[1])
	err = multierr.Append(err, dErr)
	if dErr == nil && vals["l1norm"] != test.L1Norm {
		err = multierr.Append(err, fmt.Errorf("wrong l1norm value: got %f, want %f. Table:\n%v", vals["l1norm"], test.L1Norm, tbl))
	}
	if dErr == nil && vals["l2norm"] != test.L2Norm {
		err = multierr.Append(err, fmt.Errorf("wrong l2norm value: got %f, want %f. Table:\n%v", vals["l2norm"], test.L2Norm, tbl))
	}
	return err
}

func checkDistribution(clt client.Client, path []string, test *testData) error {
	ctx := context.Background()
	dataDistPath := append(append([]string{}, path...), scopetensor.NodeNameDistribution)
	nodes, err := client.PathToNodes(ctx, clt, dataDistPath)
	if err != nil {
		return err
	}
	if err := scopetesting.CheckNodePaths(nodes); err != nil {
		return err
	}
	var wantMIME = mime.ProtoToMIME(&plotpb.Plot{})
	if err := scopetesting.CheckMIME(nodes[0].GetMime(), wantMIME); err != nil {
		return err
	}
	// Check the data returned by the server.
	data, err := client.NodesData(ctx, clt, nodes)
	if err != nil {
		return err
	}
	plt := plotpb.Plot{}
	if err := client.ToProto(data[0], &plt); err != nil {
		return err
	}
	if len(plt.Plotters) == 0 {
		return nil
	}
	if diff := cmp.Diff(&plt, test.Dist, protocmp.Transform(), tolerance(1e-5)); diff != "" {
		return fmt.Errorf("wrong distribution: %s\ngot:\n%v\nbut want:\n%v", diff, prototext.Format(&plt), test.Dist)
	}
	return nil
}

func checkTensorInfo(clt client.Client, path []string, test *testData) error {
	ctx := context.Background()
	infoPath := append(append([]string{}, path...), scopetensor.NodeNameInfo, "html")
	nodes, err := client.PathToNodes(ctx, clt, infoPath)
	if err != nil {
		return err
	}
	if err := scopetesting.CheckNodePaths(nodes); err != nil {
		return err
	}
	infoMime := mime.HTMLText
	if err := scopetesting.CheckMIME(nodes[0].GetMime(), infoMime); err != nil {
		return err
	}
	nodeData, err := client.NodesData(ctx, clt, nodes)
	if err != nil {
		return err
	}
	raw, err := client.ToRaw(nodeData[0])
	if err != nil {
		return err
	}
	got := string(raw)
	if diff := cmp.Diff(got, test.Info); diff != "" {
		return fmt.Errorf("tensor info error:\ngot: %v\nwant: %v\ndiff: %s", got, test.Info, diff)
	}
	return nil
}

// CheckTensorData checks the data provided by the server once a tensor has been written.
func CheckTensorData(clt client.Client, path []string, test *testData) error {
	if err := checkRenderedImage(clt, path, scopetensor.NodeNameImage, test.Tensor, test.ToImg()); err != nil {
		return fmt.Errorf("image error for test %q: %v", test.Desc, err)
	}
	if err := checkMetrics(clt, path, test); err != nil {
		return fmt.Errorf("metrics error: %v", err)
	}
	if err := checkDistribution(clt, path, test); err != nil {
		return fmt.Errorf("distribution error: %v", err)
	}
	if err := checkTensorInfo(clt, path, test); err != nil {
		return fmt.Errorf("tensor info error: %v", err)
	}
	if err := checkRenderedImage(clt, path, scopetensor.NodeNameBitPlane, test.Tensor, test.ToBitPlane()); err != nil {
		return fmt.Errorf("bit plane error: %v", err)
	}
	return nil
}

// CheckTensorDataAfterReset checks the data provided by the server once a tensor has been written.
func CheckTensorDataAfterReset(clt client.Client, path []string) error {
	if err := CheckTensorData(clt, path, &resetData); err != nil {
		return fmt.Errorf("incorrect data after a write reset: %v", err)
	}
	return nil
}
