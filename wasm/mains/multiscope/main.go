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

//go:build js

// Binary worker tests web-worker communication.
package main

import (
	"fmt"
	"net/url"

	pb "multiscope/protos/ui_go_proto"
	"multiscope/wasm/puller"
	"multiscope/wasm/ui/fatal"
	"multiscope/wasm/ui/uimain"
	"multiscope/wasm/version"
	"multiscope/wasm/worker"

	"honnef.co/go/js/dom/v2"

	_ "multiscope/wasm/injector/switchtheme"
	_ "multiscope/wasm/panels"
)

func grpcAddr() *pb.Connect {
	document := dom.GetWindow().Document().(dom.HTMLDocument)
	baseURI := document.BaseURI()
	uri, err := url.Parse(baseURI)
	if err != nil {
		fmt.Printf("cannot parse url %q: %v\n", baseURI, uri)
	}
	return &pb.Connect{
		Scheme: uri.Scheme,
		Host:   uri.Host,
	}
}

func pullDataMain(wkr *worker.Worker) {
	if plr := puller.New(wkr); plr != nil {
		plr.Pull()
	}
}

func main() {
	worker.Register(pullDataMain)
	if worker.IsWorker() {
		worker.Run()
		return
	}

	// Main thread.
	serverAddr := grpcAddr()
	if err := version.Check(serverAddr); err != nil {
		fatal.Display(err)
		return
	}

	// Start the data puller worker.
	pullr, err := worker.Go(pullDataMain)
	ui := uimain.NewUI(pullr, serverAddr)
	if err != nil {
		ui.DisplayErr(err)
		return
	}
	ui.MainLoop()
}
