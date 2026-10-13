# Browser Usage

The package core is runtime-neutral. In the browser (or any WebAssembly environment) install `onnxruntime-web` and pass it as `ort`; everything else is unchanged.

```bash
npm install mlconfgen onnxruntime-web
```

---

## Weights in the Browser

`egnnOnnx` and `adjMatSeerOnnx` accept a bare file name, a relative path or URL, or the model bytes:

- **Bare name** (the defaults, e.g. `"egnn_chembl_15_39.onnx"`) → fetched from `baseUrl` (the Hugging Face repo by default) and cached with the Cache API (HTTPS or localhost only; the browser may evict it). Where the Cache API is unavailable the file is fetched on every load.
- **Relative path or URL** → passed to onnxruntime-web unchanged, i.e. self-hosted weights work exactly as before.
- **`Uint8Array`** → used as is.

```js
import { createGenerator, seed } from "mlconfgen";
import * as ort from "onnxruntime-web";

// Default weights from Hugging Face, cached after the first load
const gen = await createGenerator({
  ort,
  diffusionSteps: 50,
  onProgress: ({ file, loaded, total }) => console.log(file, loaded, total),
});

seed(42);
const mols = await gen.random({ size: 4 });
```

The core weights are about 185 MB, so the first load takes a while; later loads are served from the cache.

### Self-hosted weights

Serve the ONNX files yourself and pass their paths, or point `baseUrl` at your server so that bare names are fetched (and cached) from there:

```js
// Paths, loaded by onnxruntime-web directly
const gen = await createGenerator({
  ort,
  egnnOnnx: "/models/egnn_chembl_15_39.onnx",
  adjMatSeerOnnx: "/models/adj_mat_seer_chembl_15_39.onnx",
});

// Bare names from your own server, cached with the Cache API
const gen2 = await createGenerator({ ort, baseUrl: "https://example.org/models" });
```

### Weights from the user's machine

To let users pick the ONNX files from their own machine, pass the bytes:

```js
const [egnnFile, adjFile] = fileInput.files;   // <input type="file" multiple>

const gen = await createGenerator({
  ort,
  egnnOnnx: new Uint8Array(await egnnFile.arrayBuffer()),
  adjMatSeerOnnx: new Uint8Array(await adjFile.arrayBuffer()),
});
```

`finetuneCheckpointOnnx` accepts a path or bytes in the same way.

### Cache management

`gen.listWeights()` lists the files on Hugging Face as `remote` and the files in the Cache API cache as `local`. `gen.clearCache()` deletes the cache. Use `cacheName` to keep weights of different apps on the same origin apart.

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
