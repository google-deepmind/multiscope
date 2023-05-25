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

// double is a vega example demonstrating writing to multiple writers.
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
	ticker, err := scope.NewTicker("ticker", nil)
	if err != nil {
		log.Fatal(err)
	}
	// Create two writers. Each writer will have its own plot.
	wSin, err := scope.NewScalarWriter("sin", ticker.Path())
	if err != nil {
		log.Fatal(err)
	}
	wCos, err := scope.NewScalarWriter("cos", ticker.Path())
	if err != nil {
		log.Fatal(err)
	}
	for {
		if err = ticker.Tick(); err != nil {
			break
		}
		t := float64(ticker.CurrentTick()) / 100
		if err = wSin.Write(map[string]any{
			"sa": math.Sin(t),
			"sb": math.Sin(t * 2),
			"sc": math.Sin(t / 2),
		}); err != nil {
			break
		}
		if err = wCos.Write(map[string]any{
			"ca": math.Cos(t),
			"cb": math.Cos(t * 2),
			"cc": math.Cos(t / 2),
		}); err != nil {
			break
		}
	}
	log.Fatal(err)
}
