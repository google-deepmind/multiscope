"use strict";

importScripts('/res/wasm_exec.js');
importScripts('/res/wasm.js');

onmessage = function(e) {
  console.warn('message received before the worker setup a handler:', e);
}

loadAndRunGoWasm("{{.wasmURL}}").then(() => {
  runWorker("{{.funcName}}");
});

