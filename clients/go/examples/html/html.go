// html is an example demonstrating how to write html and css to Multiscope.
package main

import (
	"flag"
	"log"
	"multiscope/clients/go/scope"
)

var (
	httpPort = flag.Int("http_port", scope.DefaultPort, "http port")
	local    = flag.Bool("local", true, "open the port to local connection only")
)

func main() {
	if err := scope.Start(*httpPort, *local); err != nil {
		log.Fatal(err)
	}
	w, err := scope.NewHTMLWriter("html", nil)
	if err != nil {
		log.Fatal(err)
	}
	if err = w.Write("<a href='http://go/multiscope/'>Multiscope</a>"); err != nil {
		log.Fatal(err)
	}
	if err = w.WriteCSS("a { color: black; }"); err != nil {
		log.Fatal(err)
	}
	<-make(chan bool)
}
