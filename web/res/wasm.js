"use strict";


async function runWASM(buffer) {
  const go = new Go();
  const result = await WebAssembly.instantiate(buffer, go.importObject);
  go.run(result.instance);
}

