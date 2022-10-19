// tensors is an example demonstrating writing tensors.
package main

import (
	"math"
	"math/rand"

	"flag"

	"log"
	"multiscope/clients/go/scope"
)

var (
	httpPort = flag.Int("http_port", scope.DefaultPort, "http port")
	local    = flag.Bool("local", true, "open the port to local connection only")
)

type tensor struct {
	shape []int
	value []float32
}

func (t *tensor) Shape() []int {
	return t.shape
}

func (t *tensor) Values() []float32 {
	return t.value
}

func newTensor(shape ...int) *tensor {
	l := 1
	for _, s := range shape {
		l *= s
	}
	return &tensor{
		shape: shape,
		value: make([]float32, l),
	}
}

func writeData(w *scope.TensorWriter, rnd *rand.Rand, t *tensor, offset float32) error {
	for i := range t.value {
		t.value[i] = rnd.Float32() + offset
	}
	return w.Write(t)
}

func writeGradient(w *scope.TensorWriter, t *tensor) error {
	dim := float32(t.Shape()[0])
	for i := range t.value {
		t.value[i] = (float32(i) / dim) - (dim / 2)
		if i > 50 && i < 55 {
			t.value[i] = float32(math.NaN())
		}
	}
	return w.Write(t)
}

func main() {
	flag.Parse()
	// Start the http server and wait forever for clients to connect.
	if err := scope.Start(*httpPort, *local); err != nil {
		log.Fatal(err)
	}
	// Create a new writer and add some data to it.
	ticker, err := scope.NewTicker("main", nil)
	if err != nil {
		log.Fatal(err)
	}
	wPos, err := scope.NewTensorWriter("Positive Data", ticker.Path())
	if err != nil {
		log.Fatal(err)
	}
	wNeg, err := scope.NewTensorWriter("Negative Data", ticker.Path())
	if err != nil {
		log.Fatal(err)
	}
	wReal, err := scope.NewTensorWriter("Real Data", ticker.Path())
	if err != nil {
		log.Fatal(err)
	}
	wGradient, err := scope.NewTensorWriter("Gradient", ticker.Path())
	if err != nil {
		log.Fatal(err)
	}
	wRGB, err := scope.NewTensorWriter("Scaled RGB", ticker.Path())
	if err != nil {
		log.Fatal(err)
	}
	tns := newTensor(20, 20)
	rgb := newTensor(20, 20, 3)
	rnd := rand.New(rand.NewSource(0))
	for {
		if err = ticker.Tick(); err != nil {
			log.Fatal(err)
		}
		if err = writeData(wPos, rnd, tns, 2); err != nil {
			log.Fatal(err)
		}
		if err = writeData(wNeg, rnd, tns, -2); err != nil {
			log.Fatal(err)
		}
		if err = writeData(wReal, rnd, tns, -.5); err != nil {
			log.Fatal(err)
		}
		if err = writeGradient(wGradient, tns); err != nil {
			log.Fatal(err)
		}
		if err = writeData(wRGB, rnd, rgb, 0); err != nil {
			log.Fatal(err)
		}
	}
}
