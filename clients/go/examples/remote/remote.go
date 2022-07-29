// double is a vega example demonstrating writing to multiple writers.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"multiscope/clients/go/remote"
	"multiscope/clients/go/scope"
	"os"
)

var servAddr = flag.String("serv_addr", "", "gRPC server address")

func main() {
	flag.Parse()
	if *servAddr == "" {
		fmt.Fprintf(os.Stderr, "no gRPC server address specified. Please use the --serv_addr flag.\n")
		os.Exit(1)
	}
	client, err := remote.Connect(context.Background(), *servAddr)
	if err != nil {
		log.Fatalf("cannot connect to multiscope: %v", err)
	}
	scope.SetGlobalClient(client)
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
