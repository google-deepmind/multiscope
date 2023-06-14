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

// Clock illustrates how to use a Multiscope clock.
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"multiscope/clients/go/examples/tensors/tensor"
	"multiscope/clients/go/remote"
	"multiscope/clients/go/scope"
)

var (
	httpPort = flag.Int("http_port", scope.DefaultPort, "http port")
	local    = flag.Bool("local", true, "open the port to local connection only")

	maxNumStepsPerEpisode = flag.Int("max_num_steps_per_episode", 10, "maximum number of steps in an episode")
	metaNumSteps          = flag.Int("meta_num_steps", math.MaxInt, "number of meta steps to run")
)

func writeTensorData(w *scope.TensorWriter, rnd *rand.Rand, t *tensor.Tensor, offset float32) error {
	vals := t.Values()
	for i := range vals {
		vals[i] = rnd.Float32() + offset
	}
	return w.Write(t)
}

func writeScalarData(w *scope.ScalarWriter, tick int) error {
	const factor = 0.01
	t := float64(tick)
	return w.Write(map[string]any{
		"a": 2 + math.Sin(t*factor),
		"b": 4 + math.Sin(t*factor),
		"c": 6 + math.Sin(t*factor),
	})
}

type writers struct {
	player *remote.Player
	text   *remote.TextWriter
	html   *remote.HTMLWriter
	scalar *remote.ScalarWriter
	rgb    *remote.TensorWriter[float32]
}

func runEpisode(ws *writers, numSteps int) error {
	const text = "Ticker\nsays\n<%d>"
	const html = `<h1 class="fancy">Ticker</h1> says <h1 class="superfancy">%d</h1>`
	rnd := rand.New(rand.NewSource(0))
	rgb := tensor.NewTensor(20, 20, 3)
	if err := ws.player.Reset(); err != nil {
		return err
	}
	for i := 0; i < numSteps; i++ {
		if err := ws.player.StoreFrame(); err != nil {
			return err
		}
		if err := ws.text.Write(fmt.Sprintf(text, ws.player.CurrentTick())); err != nil {
			return err
		}
		if err := ws.html.Write(fmt.Sprintf(html, ws.player.CurrentTick())); err != nil {
			return err
		}
		if err := writeScalarData(ws.scalar, ws.player.CurrentTick()); err != nil {
			return err
		}
		if err := writeTensorData(ws.rgb, rnd, rgb, 0); err != nil {
			return err
		}
	}
	return nil
}

func main() {
	flag.Parse()
	if err := scope.Start(*httpPort, *local); err != nil {
		log.Fatal(err)
	}
	const ignorePause = false
	meta, err := scope.NewPlayer("meta", ignorePause, nil)
	if err != nil {
		log.Fatal(err)
	}
	metaText, err := scope.NewTextWriter("Current episode", meta.Path())
	if err != nil {
		log.Fatal(err)
	}

	ws := &writers{}
	// Create a new player to store frames in it.
	if ws.player, err = scope.NewPlayer("steps", true, nil); err != nil {
		log.Fatal(err)
	}
	// Create writers under the player because StoreFrame stores data
	// for nodes under it.
	if ws.text, err = scope.NewTextWriter("TextWriter", ws.player.Path()); err != nil {
		log.Fatal(err)
	}
	if ws.html, err = scope.NewHTMLWriter("HTMLWriter", ws.player.Path()); err != nil {
		log.Fatal(err)
	}
	if err := ws.html.WriteCSS(`
	.fancy {color: var(--main-fg-color);}
	.superfancy {color: blue;}
	`); err != nil {
		log.Fatal(err)
	}
	if ws.scalar, err = scope.NewScalarWriter("Sin Data", ws.player.Path()); err != nil {
		log.Fatal(err)
	}
	if ws.rgb, err = scope.NewTensorWriter[float32]("Scaled RGB", ws.player.Path()); err != nil {
		log.Fatal(err)
	}
	rnd := rand.New(rand.NewSource(0))
	for episode := 0; episode < *metaNumSteps; episode++ {
		numSteps := rnd.Intn(*maxNumStepsPerEpisode-2) + 2
		if err := metaText.Write(fmt.Sprintf("Episode: %d\nNum steps: %d", episode, numSteps)); err != nil {
			log.Fatal(err)
		}
		if err := runEpisode(ws, numSteps); err != nil {
			log.Fatal(err)
		}
		if err := meta.StoreFrame(); err != nil {
			log.Fatal(err)
		}

	}
	<-make(chan bool)
}
