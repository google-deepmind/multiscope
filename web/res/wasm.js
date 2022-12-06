"use strict";

async function loadAndRunGoWasm(wasmURL) {
  const go = new Go();
  const buffer = pako.ungzip(await (await fetch(wasmURL)).arrayBuffer());
  const result = await WebAssembly.instantiate(buffer, go.importObject);
  go.run(result.instance);
}

