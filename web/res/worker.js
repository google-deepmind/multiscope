"use strict";

importScripts('/res/wasm_exec.js');
importScripts('/res/wasm.js');
importScripts('https://cdnjs.cloudflare.com/ajax/libs/pako/2.0.2/pako.min.js')

onmessage = function(e) {
  console.warn('message received before the worker setup a handler:', e);
}

loadAndRunGoWasm("{{.wasmURL}}").then(() => {
  runWorker("{{.funcName}}");
});

