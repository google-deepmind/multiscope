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

package worker

import (
	"fmt"
	pb "multiscope/protos/ui_go_proto"
	"syscall/js"
)

// IsWorker returns true if the code is executed from a web worker.
func IsWorker() bool {
	return !js.Global().Get("document").Truthy()
}

func run(this js.Value, args []js.Value) any {
	if len(args) == 0 {
		panic("no function name given to the web worker. Please use runWorker(functionName);")
	}
	name := args[0].String()
	f, ok := registry[name]
	if !ok {
		panic(fmt.Errorf("no function with name %q registered", name))
	}
	go func() {
		wkr := newWorker("worker:"+name, js.Global())
		m := &pb.WorkerAck{}
		if err := wkr.Send(m, nil); err != nil {
			panic(fmt.Errorf("ERROR in worker: cannot send acknowledgement to main: %v", err))
		}
		if _, err := wkr.Recv(m); err != nil {
			panic(fmt.Errorf("ERROR in worker: cannot receive acknowledgement from main: %v", err))
		}
		f(wkr)
	}()
	return nil
}

func sendWASMToWorker(wkr *Worker) error {
	wasmBuffer := js.Global().Get("globalWASMBuffer")
	wkr.val.Call("postMessage", map[string]any{
		"type":   "wasmbuffer",
		"buffer": wasmBuffer,
	})
	return nil
}

// Go starts a function in a web worker.
func Go(f MainFunc) (wkr *Worker, err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("cannot start worker from function %q: %w", f.name(), err)
		}
	}()
	workers := js.Global().Get("Worker")
	val := workers.New("worker/" + f.name())
	wkr = newWorker("main", val)
	if err := sendWASMToWorker(wkr); err != nil {
		return nil, fmt.Errorf("cannot send WASM buffer to worker: %v", err)
	}
	m := &pb.WorkerAck{}
	if _, err := wkr.Recv(m); err != nil {
		return nil, fmt.Errorf("cannot receive acknowledgement from worker: %v", err)
	}
	if err := wkr.Send(m, nil); err != nil {
		return nil, fmt.Errorf("cannot send acknowledgement from main: %v", err)
	}
	return wkr, nil
}

// Run registers a javascript workerRun function and wait for that function to be called.
// This function never returns.
func Run() {
	js.Global().Set("runWorker", js.FuncOf(run))
	<-make(chan bool)
}
