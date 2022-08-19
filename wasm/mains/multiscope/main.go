//go:build js

// Binary worker tests web-worker communication.
package main

import (
	"fmt"
	"net/url"

	pb "multiscope/protos/ui_go_proto"
	"multiscope/wasm/puller"
	"multiscope/wasm/ui/uimain"
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
	puller.New(wkr).Pull()
}

func main() {
	worker.Register(pullDataMain)
	if worker.IsWorker() {
		worker.Run()
		return
	}

	// Main thread.

	// Start the data puller worker.
	puller, err := worker.Go(pullDataMain)
	ui := uimain.NewUI(puller, grpcAddr())
	if err != nil {
		ui.DisplayErr(err)
		return
	}
	ui.MainLoop()
}
