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
	"context"
	"fmt"
	"net/url"

	"multiscope/internal/httpgrpc"
	"multiscope/internal/version"
	treepb "multiscope/protos/tree_go_proto"
	uipb "multiscope/protos/ui_go_proto"
	"multiscope/wasm/puller"
	"multiscope/wasm/ui/fatal"
	"multiscope/wasm/ui/uimain"
	"multiscope/wasm/worker"

	"github.com/pkg/errors"
	"honnef.co/go/js/dom/v2"

	_ "multiscope/wasm/injector/switchtheme"
	_ "multiscope/wasm/panels"
)

func grpcAddr() *uipb.Connect {
	document := dom.GetWindow().Document().(dom.HTMLDocument)
	baseURI := document.BaseURI()
	uri, err := url.Parse(baseURI)
	if err != nil {
		fmt.Printf("cannot parse url %q: %v\n", baseURI, err)
	}
	return &uipb.Connect{
		Scheme: uri.Scheme,
		Host:   uri.Host,
		RawUrl: baseURI,
	}
}

// Check the version of the gRPC API.
// Returns an error if it does not match.
func getTreeID(addr *uipb.Connect) (*treepb.TreeID, error) {
	conn := httpgrpc.Connect(addr.Scheme, addr.Host)
	treeClient := treepb.NewTreeClient(conn)
	req := &treepb.GetTreeIDRequest{Url: addr.RawUrl}
	resp, err := treeClient.GetTreeID(context.Background(), req)
	if err != nil {
		return nil, errors.Errorf("cannot get a tree ID: %v", err)
	}
	if resp.Version != version.Version {
		return nil, errors.Errorf("frontend client version %q does not match the server gRPC API version %q.\nYou may need to recompile the WASM frontend client, restart the server, then reload the page", resp.Version, version.Version)
	}
	return resp.TreeId, nil
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
	var err error
	if serverAddr.TreeId, err = getTreeID(serverAddr); err != nil {
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
