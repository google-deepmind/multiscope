"use strict";

importScripts('/res/wasm_exec.js');
importScripts('/res/wasm.js');

onmessage = function(ev) {
  const data = ev.data;
  if (data.type != "wasmbuffer") {
    console.warn('message received before the worker has been setup:', e);
    return;
  }
  runWASM(data.buffer).then(function () {
    runWorker("{{.funcName}}");
  });
}

