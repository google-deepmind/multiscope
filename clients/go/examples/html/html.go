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
	flag.Parse()
	if err := scope.Start(*httpPort, *local); err != nil {
		log.Fatal(err)
	}
	w, err := scope.NewHTMLWriter("html", nil)
	if err != nil {
		log.Fatal(err)
	}
	if err = w.Write("<a href='https://www.google.com/search?q=multiscope'>Multiscope</a>"); err != nil {
		log.Fatal(err)
	}
	if err = w.WriteCSS("a { color: black; }"); err != nil {
		log.Fatal(err)
	}
	<-make(chan bool)
}
