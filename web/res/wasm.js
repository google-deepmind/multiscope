"use strict";

function loadAndRunGoWasm(wasmURL) {
  const go = new Go();
  return WebAssembly.instantiateStreaming(fetch(wasmURL), go.importObject).then(
    (result) => {
      go.run(result.instance);
    }
  );
}

