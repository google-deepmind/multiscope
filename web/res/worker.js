"use strict";

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

