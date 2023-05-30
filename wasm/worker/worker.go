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

// Package worker provides an API to use web workers.
package worker

import (
	"fmt"
	"syscall/js"

	"github.com/pkg/errors"
	"google.golang.org/protobuf/proto"
	"honnef.co/go/js/dom/v2"
)

// Worker is a web worker.
type Worker struct {
	name string
	val  js.Value
	msgs chan js.Value
	buf  js.Value
}

func newWorker(name string, val js.Value) *Worker {
	w := &Worker{
		name: name,
		val:  val,
		msgs: make(chan js.Value, 1),
	}
	w.val.Set("onmessage", js.FuncOf(func(this js.Value, args []js.Value) any {
		if len(args) != 1 {
			panic("worker.onmessage received the wrong number of arguments")
		}
		data := dom.WrapEvent(args[0]).(*dom.MessageEvent).Data()
		go func() {
			w.msgs <- data
		}()
		return nil
	}))
	return w
}

// Send a message to the web worker or to the main worker.
// The function returns immediately.
func (w *Worker) Send(m proto.Message, aux any, transferables ...any) error {
	buf, err := proto.Marshal(m)
	if err != nil {
		return fmt.Errorf("cannot marshal message: %w", err)
	}
	if !w.buf.Truthy() || w.buf.Length() != len(buf) {
		w.buf = js.Global().Get("Uint8Array").New(len(buf))
	}
	js.CopyBytesToJS(w.buf, buf)
	typeURL := string(proto.MessageName(m))
	jsTransferables := js.ValueOf(transferables)
	w.val.Call("postMessage", map[string]any{
		"proto":   w.buf,
		"size":    len(buf),
		"typeurl": typeURL,
		"aux":     aux,
	}, jsTransferables)
	return nil
}

// Recv receives a message from another webworker.
func (w *Worker) Recv(m proto.Message) (js.Value, error) {
	data := <-w.msgs
	want := string(proto.MessageName(m))
	got := data.Get("typeurl").String()
	if got != want {
		return js.Null(), errors.Errorf("cannot receive next message: got type %q but want type %q", got, want)
	}
	size := data.Get("size").Int()
	if !w.buf.Truthy() || w.buf.Length() != size {
		w.buf = js.Global().Get("Uint8Array").New(size)
	}
	buf := make([]uint8, size)
	js.CopyBytesToGo(buf, data.Get("proto"))
	if err := proto.Unmarshal(buf, m); err != nil {
		return js.Null(), fmt.Errorf("cannot unmarshal data of type %q to proto of type %q: %v", got, want, err)
	}
	return data.Get("aux"), nil
}

// Close the worker.
func (w *Worker) Close() {
	close(w.msgs)
	w.msgs = nil
}
