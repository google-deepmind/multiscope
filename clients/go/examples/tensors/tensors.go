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

// tensors is an example demonstrating writing tensors.
package main

import (
	"math"
	"math/rand"

	"flag"

	"log"
	"multiscope/clients/go/examples/tensors/tensor"
	"multiscope/clients/go/scope"
)

var (
	httpPort = flag.Int("http_port", scope.DefaultPort, "http port")
	local    = flag.Bool("local", true, "open the port to local connection only")
)

func writeData(w *scope.TensorWriter, rnd *rand.Rand, t *tensor.Tensor, offset float32) error {
	vals := t.Values()
	for i := range vals {
		vals[i] = rnd.Float32() + offset
	}
	return w.Write(t)
}

func writeGradient(w *scope.TensorWriter, t *tensor.Tensor) error {
	dim := float32(t.Shape()[0])
	vals := t.Values()
	for i := range vals {
		vals[i] = (float32(i) / dim) - (dim / 2)
		if i > 50 && i < 55 {
			vals[i] = float32(math.NaN())
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
	wPos, err := scope.NewTensorWriter[float32]("Positive Data", ticker.Path())
	if err != nil {
		log.Fatal(err)
	}
	wNeg, err := scope.NewTensorWriter[float32]("Negative Data", ticker.Path())
	if err != nil {
		log.Fatal(err)
	}
	wReal, err := scope.NewTensorWriter[float32]("Real Data", ticker.Path())
	if err != nil {
		log.Fatal(err)
	}
	wGradient, err := scope.NewTensorWriter[float32]("Gradient", ticker.Path())
	if err != nil {
		log.Fatal(err)
	}
	wRGB, err := scope.NewTensorWriter[float32]("Scaled RGB", ticker.Path())
	if err != nil {
		log.Fatal(err)
	}
	tns := tensor.NewTensor(20, 20)
	rgb := tensor.NewTensor(20, 20, 3)
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
