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
	wSin, err := scope.NewScalarWriter("Sin", nil)
	if err != nil {
		log.Fatal(err)
	}
	wCos, err := scope.NewScalarWriter("Cos", nil)
	if err != nil {
		log.Fatal(err)
	}
	for {
		if err = ticker.Tick(); err != nil {
			break
		}
		t := float64(ticker.CurrentTick()) / 100
		if err = wSin.Write(map[string]interface{}{
			"sa": math.Sin(t),
			"sb": math.Sin(t * 2),
			"sc": math.Sin(t / 2),
		}); err != nil {
			break
		}
		if err = wCos.Write(map[string]interface{}{
			"ca": math.Cos(t),
			"cb": math.Cos(t * 2),
			"cc": math.Cos(t / 2),
		}); err != nil {
			break
		}
	}
	log.Fatal(err)
}
