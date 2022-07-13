// Clock illustrates how to use a Multiscope clock.
package main

import (
	"flag"
	"fmt"
	"log"
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
	w1, err := scope.NewTextWriter("TextWriter", ticker.Path())
	if err != nil {
		log.Fatal(err)
	}
	w2, err := scope.NewHTMLWriter("HTMLWriter", ticker.Path())
	if err != nil {
		log.Fatal(err)
	}
	w2.WriteCSS(`
	.fancy {color: red;}
	.superfancy {color: blue;}
	`)
	const text = "Ticker\nsays\n<%d>"
	const html = `<h1 class="fancy">Ticker</h1> says <h1 class="superfancy">%d</h1>`
	for {
		if err = ticker.Tick(); err != nil {
			break
		}
		if err = w1.Write(fmt.Sprintf(text, ticker.CurrentTick())); err != nil {
			break
		}
		if err = w2.Write(fmt.Sprintf(html, ticker.CurrentTick())); err != nil {
			break
		}
	}
	log.Fatal(err)
}
