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
	"fmt"
	"io/ioutil"
	"log"
	"multiscope/clients/go/html"
	"multiscope/clients/go/scope"
	"multiscope/internal/fmtx"
	"os"
	"path"
	"sort"
)

var (
	httpPort = flag.Int("http_port", scope.DefaultPort, "http port")
	local    = flag.Bool("local", true, "open the port to local connection only")
)

func writeDirContent(w *scope.HTMLWriter, dirname string) error {
	files, err := ioutil.ReadDir(dirname)
	if err != nil {
		return fmt.Errorf("cannot read directory %v: %w", dirname, err)
	}
	sort.Slice(files, func(i, j int) bool {
		return files[i].Name() < files[i].Name()
	})
	cnt := html.NewContent(w)
	up := html.NewButton(cnt, html.HTML("..")).OnClick(func() error {
		return writeDirContent(w, path.Dir(dirname))
	})
	cnt.Append(up, html.BR)
	for _, file := range files {
		if !file.IsDir() {
			cnt.Append(html.HTML(file.Name()), html.BR)
			continue
		}
		fileName := file.Name()
		button := html.NewButton(cnt, html.HTML(fileName)).OnClick(func() error {
			return writeDirContent(w, path.Join(dirname, fileName))
		})
		cnt.Append(button, html.BR)
	}
	return w.WriteStringer(cnt)
}

func main() {
	flag.Parse()
	if err := scope.Start(*httpPort, *local); err != nil {
		log.Fatal(err)
	}
	w, err := scope.NewHTMLWriter("html", nil)
	if err != nil {
		log.Fatal(fmtx.FormatError(err))
	}
	dirname, err := os.UserHomeDir()
	if err != nil {
		log.Fatalf("cannot get user home directory: %v", err)
	}
	if err := writeDirContent(w, dirname); err != nil {
		log.Fatal(fmtx.FormatError(err))
	}
	<-make(chan bool)
}
