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
	"multiscope/internal/server/writers/tensor"
	scopetesting "multiscope/internal/testing"
	plotpb "multiscope/protos/plot_go_proto"
	pb "multiscope/protos/tree_go_proto"
	pbgrpc "multiscope/protos/tree_go_proto"

	"github.com/google/go-cmp/cmp"
	"go.uber.org/multierr"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/testing/protocmp"
)

type tensorImpl struct {
	shape []int
	value []float32
}

// Shape returns the shape of the testing tensor.
func (t *tensorImpl) Shape() []int {
	return t.shape
}

// Values returns the set of values of the testing tensor.
func (t *tensorImpl) Values() []float32 {
	return t.value
}

// Tensor01Name is the name of the tensor writer in the tree.
const Tensor01Name = "tensor01"

type testData struct {
	Desc               string
	Tensor             tensor.Tensor
	ToImg, ToBitPlane  func() *image.RGBA
	Min, Max           float32
	ScaleMin, ScaleMax float32
	L1Norm, L2Norm     float32
	Dist               *plotpb.Plot
	Info               string
}

func distributionToSerie(dist [][]float32) *plotpb.Serie {
	serie := &plotpb.Serie{}
	if len(dist) == 0 {
		return serie
	}
	for _, row := range dist {
		serie.Points = append(serie.Points, &plotpb.Point{
			X: float64(row[0]), Y: float64(row[1]),
		})
	}
	return serie
}

func buildDistribution(dist [][]float32) *plotpb.Plot {
	return &plotpb.Plot{
		Plotters: []*plotpb.Plotter{
			{
				Serie: distributionToSerie(dist),
			},
		},
	}
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
			Dist: buildDistribution([][]float32{
				{1.125, 1},
				{1.375, 0},
				{1.625, 0},
				{1.875, 0},
				{2.125, 1},
				{2.375, 0},
				{2.625, 0},
				{2.875, 0},
				{3.125, 1},
				{3.375, 0},
				{3.625, 0},
				{3.875, 0},
				{4.125, 1},
				{4.375, 0},
				{4.625, 0},
				{4.875, 0},
				{5.125, 1},
				{5.375, 0},
				{5.625, 0},
				{5.875, 1},
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
			Dist: buildDistribution([][]float32{
				{-5.875, 1},
				{-5.625, 0},
				{-5.375, 0},
				{-5.125, 0},
				{-4.875, 1},
				{-4.625, 0},
				{-4.375, 0},
				{-4.125, 0},
				{-3.875, 1},
				{-3.625, 0},
				{-3.375, 0},
				{-3.125, 0},
				{-2.875, 1},
				{-2.625, 0},
				{-2.375, 0},
				{-2.125, 0},
				{-1.875, 1},
				{-1.625, 0},
				{-1.375, 0},
				{-1.125, 1},
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
			Dist: buildDistribution([][]float32{
				{-5.725, 1},
				{-5.1749997, 0},
				{-4.6249995, 0},
				{-4.0749993, 0},
				{-3.5249994, 0},
				{-2.9749994, 0},
				{-2.4249995, 0},
				{-1.8749995, 1},
				{-1.3249996, 0},
				{-0.77499956, 0},
				{-0.22499955, 1},
				{0.32500046, 0},
				{0.8750005, 1},
				{1.4250004, 0},
				{1.9750004, 0},
				{2.5250003, 0},
				{3.0750003, 1},
				{3.6250002, 0},
				{4.175, 0},
				{4.7250004, 1},
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
func NewTestTensor() tensor.Tensor {
	return &tensorImpl{
		shape: []int{1, 2, 3},
		value: []float32{-6, 5, -4, 3, -2, 1},
	}
}

// checkRenderedImage checks the data that can be read from the tensor01 node.
func checkRenderedImage(clt pbgrpc.TreeClient, path []string, nodeName string, tns tensor.Tensor, want *image.RGBA) error {
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
	if len(tns.Values()) == 0 {
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

func extractPlotData(data *pb.NodeData) (*plotpb.Plot, map[string]float32, error) {
	plt := plotpb.Plot{}
	if err := client.ToProto(data, &plt); err != nil {
		return nil, nil, err
	}
	return extractDataFromPlot(&plt)
}

func extractScalarPlotData(data *pb.NodeData) (*plotpb.Plot, map[string]float32, error) {
	plt := plotpb.ScalarsPlot{}
	if err := client.ToProto(data, &plt); err != nil {
		return nil, nil, err
	}
	return extractDataFromPlot(plt.Plot)
}

func extractDataFromPlot(plt *plotpb.Plot) (*plotpb.Plot, map[string]float32, error) {
	vals := make(map[string]float32)
	for _, plotter := range plt.Plotters {
		serie := plotter.Serie
		vals[plotter.Legend] = float32(serie.Points[len(serie.Points)-1].Y)
	}
	return plt, vals, nil
}

func checkMetrics(clt pbgrpc.TreeClient, path []string, test *testData) error {
	ctx := context.Background()
	paths := [][]string{
		append(append([]string{}, path...), tensor.NodeNameMinMax),
		append(append([]string{}, path...), tensor.NodeNameNorms),
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

func checkDistribution(clt pbgrpc.TreeClient, path []string, test *testData) error {
	ctx := context.Background()
	dataDistPath := append(append([]string{}, path...), tensor.NodeNameDistribution)
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
	if diff := cmp.Diff(&plt, test.Dist, protocmp.Transform()); diff != "" {
		return fmt.Errorf("wrong distribution: %s\ngot:\n%v\nbut want:\n%v", diff, prototext.Format(&plt), test.Dist)
	}
	return nil
}

func checkTensorInfo(clt pbgrpc.TreeClient, path []string, test *testData) error {
	ctx := context.Background()
	infoPath := append(append([]string{}, path...), tensor.NodeNameInfo, "html")
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
func CheckTensorData(clt pbgrpc.TreeClient, path []string, test *testData) error {
	if err := checkRenderedImage(clt, path, tensor.NodeNameImage, test.Tensor, test.ToImg()); err != nil {
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
	if err := checkRenderedImage(clt, path, tensor.NodeNameBitPlane, test.Tensor, test.ToBitPlane()); err != nil {
		return fmt.Errorf("bit plane error: %v", err)
	}
	return nil
}

// CheckTensorDataAfterReset checks the data provided by the server once a tensor has been written.
func CheckTensorDataAfterReset(clt pbgrpc.TreeClient, path []string) error {
	if err := CheckTensorData(clt, path, &resetData); err != nil {
		return fmt.Errorf("incorrect data after a write reset: %v", err)
	}
	return nil
}
