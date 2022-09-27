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
