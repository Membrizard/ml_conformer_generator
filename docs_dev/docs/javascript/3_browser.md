# Browser Usage

The package core is runtime-neutral. In the browser (or any WebAssembly environment) install `onnxruntime-web` and pass it as `ort`; everything else is unchanged.

```bash
npm install mlconfgen onnxruntime-web
```

---

## Loading Weights Client-Side

The weights are not published on npm. A common pattern is to let the user pick the ONNX files from their own machine so that they are never uploaded or re-hosted:

```js
import { createGenerator, seed } from "mlconfgen";
import * as ort from "onnxruntime-web";

const [egnnFile, adjFile] = fileInput.files;   // <input type="file" multiple>

const gen = await createGenerator({
  ort,
  egnnOnnx: new Uint8Array(await egnnFile.arrayBuffer()),
  adjMatSeerOnnx: new Uint8Array(await adjFile.arrayBuffer()),
  diffusionSteps: 50,
});

seed(42);
const mols = await gen.generateConformers({
  referenceContext: [89.87, 210.78, 217.78],
  nAtoms: 20,
  nSamples: 4,
});
```

`egnnOnnx`, `adjMatSeerOnnx` and `finetuneCheckpointOnnx` accept `Uint8Array` buffers in place of paths. Buffers fetched from your own server (`fetch(url).then(r => r.arrayBuffer())`) work the same way.

---

## RDKit in the Browser

The bundled `@rdkit/rdkit` is resolved automatically for the runtime it finds. Three setups are supported, tried in order:

1. A global `initRDKitModule` defined by a plain `<script>` tag.
2. A bundler (Vite, webpack, …) that interops the CommonJS build into a usable `default` export.
3. No bundler — the UMD build is injected as a classic script from the jsDelivr CDN.

To pin a specific RDKit build (e.g. a self-hosted WASM), pass `rdkitLoader`:

```js
const gen = await createGenerator({
  ort,
  egnnOnnx,
  adjMatSeerOnnx,
  rdkitLoader: () =>
    initRDKitModule({ locateFile: () => "/wasm/RDKit_minimal.wasm" }),
});
```

The loader is module-global: calling `createGenerator` without `rdkitLoader` resets it to the bundled default.

### Failure modes

- If RDKit is configured but fails to initialise (e.g. the `.wasm` cannot be fetched), generation throws an error named `RdkitLoadError` rather than silently reporting every molecule as invalid.
- Inside a Web Worker there is no DOM to inject a script into; pass an explicit `rdkitLoader` (e.g. using `importScripts`).

---

## Threads and Performance

- `onnxruntime-web` runs on WebAssembly by default; enable multi-threading and SIMD through `ort.env.wasm` (e.g. `ort.env.wasm.numThreads = 4`) before creating the generator. Multi-threading requires cross-origin isolation headers.
- The WebGPU execution provider can be selected through the `onnxruntime-web` session options if your build supports it; the package passes model paths/bytes straight to `ort.InferenceSession.create`.
- Keep `nSamples` small per call in the browser (`1–10`) and prefer `diffusionSteps` of `20–50` for interactive use.
- Run generation in a Web Worker to keep the UI responsive; all package APIs are `async` and worker-safe given an explicit `rdkitLoader`.

---

## Smoke Viewer

The repository includes a minimal browser trajectory viewer (`js/smoke/trajectory_viewer.html`) served by `npm run smoke`, which demonstrates streaming frames from [`animateGeneration`](4_animation.md) into a 3D view.
