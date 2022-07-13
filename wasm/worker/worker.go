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
		w.msgs <- dom.WrapEvent(args[0]).(*dom.MessageEvent).Data()
		return nil
	}))
	return w
}

// PostMessage sends a message to the web worker or to the main worker.
// The function returns immediatly.
func (w *Worker) Send(m proto.Message, aux any, transferables ...any) error {
	buf, err := proto.Marshal(m)
	if err != nil {
		return fmt.Errorf("cannot marshal message: %w", err)
	}
	if !w.buf.Truthy() || w.buf.Length() != len(buf) {
		w.buf = js.Global().Get("Uint8Array").New(len(buf))
	}
	js.CopyBytesToJS(w.buf, buf)
	w.val.Call("postMessage", map[string]any{
		"proto":   w.buf,
		"size":    len(buf),
		"typeurl": string(proto.MessageName(m)),
		"aux":     aux,
	}, js.ValueOf(transferables))
	return nil
}

// OnMessage registers a callback for when the worker receives a message.
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
