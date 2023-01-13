// Clock illustrates how to use a Multiscope clock.
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"multiscope/clients/go/examples/tensors/tensor"
	"multiscope/clients/go/scope"
)

var (
	httpPort = flag.Int("http_port", scope.DefaultPort, "http port")
	local    = flag.Bool("local", true, "open the port to local connection only")
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

func main() {
	flag.Parse()
	if err := scope.Start(*httpPort, *local); err != nil {
		log.Fatal(err)
	}
	// Create a new player to store frames in it.
	player, err := scope.NewPlayer("player", true, nil)
	if err != nil {
		log.Fatal(err)
	}
	// Create writers under the player because StoreFrame stores data
	// for nodes under it.
	wText, err := scope.NewTextWriter("TextWriter", player.Path())
	if err != nil {
		log.Fatal(err)
	}
	wHTML, err := scope.NewHTMLWriter("HTMLWriter", player.Path())
	if err != nil {
		log.Fatal(err)
	}
	err = wHTML.WriteCSS(`
	.fancy {color: var(--main-fg-color);}
	.superfancy {color: blue;}
	`)
	if err != nil {
		log.Fatal(err)
	}
	wScalar, err := scope.NewScalarWriter("Sin Data", player.Path())
	if err != nil {
		log.Fatal(err)
	}
	wRGB, err := scope.NewTensorWriter("Scaled RGB", player.Path())
	if err != nil {
		log.Fatal(err)
	}
	const text = "Ticker\nsays\n<%d>"
	const html = `<h1 class="fancy">Ticker</h1> says <h1 class="superfancy">%d</h1>`
	rnd := rand.New(rand.NewSource(0))
	rgb := tensor.NewTensor(20, 20, 3)
	for {
		if err = player.StoreFrame(); err != nil {
			break
		}
		if err = wText.Write(fmt.Sprintf(text, player.CurrentTick())); err != nil {
			break
		}
		if err = wHTML.Write(fmt.Sprintf(html, player.CurrentTick())); err != nil {
			break
		}
		if err = writeScalarData(wScalar, player.CurrentTick()); err != nil {
			break
		}
		if err = writeTensorData(wRGB, rnd, rgb, 0); err != nil {
			break
		}
	}
	log.Fatal(err)
}