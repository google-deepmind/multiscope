// Package workerserver provides web-worker on the server side.
package workerserver

import (
	"fmt"
	"log"
	"net/http"
	"path"
	"text/template"
)

const script = `
importScripts('/res/wasm_exec.js');

onmessage = function(e) {
  console.warn('message received before the worker setup a handler:', e);
}

const goWorker = new Go();
WebAssembly.instantiateStreaming(
  fetch("/res/{{.wasmName}}.wasm"),
  goWorker.importObject
).then((result) => {
  goWorker.run(result.instance);
  runWorker("{{.funcName}}");
});
`

func wErr(w http.ResponseWriter, format string, a ...interface{}) {
	fmt.Println("error:", fmt.Sprintf(format, a...))
	_, err := w.Write([]byte(fmt.Sprintf(format, a...)))
	if err != nil {
		log.Printf("error sending the error back to the client: %v", err)
	}
}

func split(p string) (wasmName, funcName string, err error) {
	var dir string
	dir, funcName = path.Split(p)
	if dir == "" || len(dir) <= 1 {
		return "", "", fmt.Errorf("empty wasm name")
	}
	_, wasmName = path.Split(dir[:len(dir)-1])
	if wasmName == "" {
		return "", "", fmt.Errorf("empty wasm name")
	}
	return
}

// Handle requests to start new web workers.
func Handle(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("content-type", "application/javascript")
	t := template.New("main")
	wasmName, funcName, err := split(r.URL.Path)
	if err != nil {
		wErr(w, "cannot get wasm file and function name from path %q: %v. Expected format is wasm_name/func_name", r.URL.Path, err)
		return

	}
	t, err = t.Parse(script)
	if err != nil {
		wErr(w, "error parsing template: %v", err)
		return
	}
	args := map[string]string{
		"wasmName": wasmName,
		"funcName": funcName,
	}
	if err = t.Execute(w, args); err != nil {
		wErr(w, "error writing content: %v", err)
		return
	}
}
