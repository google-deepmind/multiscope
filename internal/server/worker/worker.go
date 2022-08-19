// Package worker provides web-worker on the server side.
package worker

import (
	"fmt"
	"log"
	"multiscope/internal/server/wasmurl"
	"net/http"
	"strings"
	"text/template"
)

const script = `
importScripts('/res/wasm_exec.js');

onmessage = function(e) {
  console.warn('message received before the worker setup a handler:', e);
}

const goWorker = new Go();
WebAssembly.instantiateStreaming(
  fetch("{{.wasmURL}}"),
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

func parseFuncName(s string) (string, error) {
	prefix, funcName, found := strings.Cut(s, "/")
	if prefix == "" {
		prefix, funcName, found = strings.Cut(funcName, "/")
	}
	if !found {
		return "", fmt.Errorf("malformed URL %q: cannot find separator '/'", s)
	}
	if prefix != "worker" {
		return "", fmt.Errorf("URL has the wrong prefix: got %q but want worker", prefix)
	}
	return funcName, nil
}

// Handle requests to start new web workers.
func Handle(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("content-type", "application/javascript")
	funcName, err := parseFuncName(r.URL.Path)
	if err != nil {
		wErr(w, "cannot provide worker: %v", err)
		return
	}
	t := template.New("main")
	t, err = t.Parse(script)
	if err != nil {
		wErr(w, "error parsing template: %v", err)
		return
	}
	args := map[string]string{
		"funcName": funcName,
		"wasmURL":  wasmurl.URL(),
	}
	if err = t.Execute(w, args); err != nil {
		wErr(w, "error writing content: %v", err)
		return
	}
}
