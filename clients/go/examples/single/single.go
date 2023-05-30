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

// single is a vega example demonstrating writing to multiple writers.
package main

import (
	"flag"
	"log"
	"math"
	"multiscope/clients/go/scope"
)

var (
	httpPort = flag.Int("http_port", scope.DefaultPort, "http port")
	local    = flag.Bool("local", true, "open the port to local connection only")
)

func main() {
	flag.Parse()
	if err := scope.Start(*httpPort, *local); err != nil {
		log.Fatal(err)
	}
	// Create a new writer and add some data to it.
	ticker, err := scope.NewTicker("main", nil)
	if err != nil {
		log.Fatal(err)
	}
	w, err := scope.NewScalarWriter("Sin Data", ticker.Path())
	if err != nil {
		log.Fatal(err)
	}
	for {
		if err = ticker.Tick(); err != nil {
			break
		}
		const factor = 0.01
		t := float64(ticker.CurrentTick())
		if err = w.Write(map[string]any{
			"a": 2 + math.Sin(t*factor),
			"b": 4 + math.Sin(t*factor),
			"c": 6 + math.Sin(t*factor),
		}); err != nil {
			break
		}
	}
	log.Fatal(err)
}
