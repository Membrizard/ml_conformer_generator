# ML Conformer Generator JS (JavaScript / ONNX Runtime)

JavaScript package for spatially-aware molecule generation via equivariant diffusion.
Bundles **@rdkit/rdkit**; you bring your own ONNX Runtime.

<img src="https://raw.githubusercontent.com/Membrizard/ml_conformer_generator/main/assets/logo/mlconfgen_logo.png" width="120" style="display: block; margin: 0 10%;">

## Install

```bash
npm install mlconfgen onnxruntime-node
```

`onnxruntime-node` is an **optional peer dependency** — the package core is runtime-neutral
and you pass the runtime in. Install `onnxruntime-node` for Node, or `onnxruntime-web`
for the browser / a WebAssembly build, and pass whichever you installed as `ort` (see below).
`@rdkit/rdkit` ships with the package, so there's nothing else to install.

For local development from this repository:

```bash
cd js
npm install
```

Model weights are not part of the npm package. They are downloaded from Hugging Face on
first use and cached — see [Model weights](#model-weights).

## Quick start

```js
import { createGenerator, seed } from "mlconfgen";
import * as ort from "onnxruntime-node";

seed(42);

// Default weights, downloaded to ~/.mlconfgen_weights on first run
const gen = await createGenerator({ ort, diffusionSteps: 100 });

const mols = await gen.generateConformers({
  referenceContext: [89.87, 210.78, 217.78], // MOI eigenvalues
  nAtoms: 20,
  nSamples: 10,
  variance: 2,
});

for (const mol of mols) {
  console.log(mol.toMolBlock());
}
```

`createGenerator(options)` takes the ONNX Runtime namespace as the `ort` option.

## Model weights

| File | Model |
|---|---|
| `egnn_chembl_15_39.onnx` | EGNN denoiser (core) — default `egnnOnnx` |
| `adj_mat_seer_chembl_15_39.onnx` | AdjMatSeer (core) — default `adjMatSeerOnnx` |
| `small_egnn_chembl_15_39.onnx` | EGNN denoiser (distilled) |
| `small_adj_mat_seer_obabel_15_39.onnx` | AdjMatSeer (distilled, OpenBabel bond perception) |

The weights are hosted at
[huggingface.co/Membrizard/ml_conformer_generator](https://huggingface.co/Membrizard/ml_conformer_generator)
and licensed under Apache 2.0 (see
[`LICENSE-MODEL`](https://github.com/Membrizard/ml_conformer_generator/blob/main/LICENSE-MODEL)).

`egnnOnnx` and `adjMatSeerOnnx` accept a file path, a bare file name or the model bytes
(`Uint8Array`). In Node a string is resolved in the same order as in the Python package:

1. an existing file path is used as is;
2. otherwise a bare file name is looked up in the cache — `~/.mlconfgen_weights`, the
   same directory the Python package uses;
3. if it is not cached it is downloaded from Hugging Face into the cache. Concurrent
   downloads of the same file are guarded by a lock file shared with Python.

Two deliberate differences from Python: a path that is not a bare name and does not exist
fails at once (`Weights file not found`) instead of being looked up on Hugging Face, and
only `*.onnx` file names count as bare names.

```js
// Distilled weights, by name
const gen = await createGenerator({
  ort,
  egnnOnnx: "small_egnn_chembl_15_39.onnx",
  adjMatSeerOnnx: "small_adj_mat_seer_obabel_15_39.onnx",
  onProgress: ({ file, loaded, total }) => console.log(file, loaded, total),
});

await gen.listWeights(); // { remote: [...], local: [...] } — .onnx files
await gen.clearCache();  // removes everything in the cache
```

`listWeights()` always lists the Hugging Face repo as `remote`; if that request fails,
`remote` is empty and `remoteError` holds the error, while `local` is still filled in.

Options for the cache and download location (on `createGenerator` or `WeightsManager`):

- `cacheDir` — cache directory (Node). Defaults to the `MLCONFGEN_CACHE` environment
  variable (JS only), then `~/.mlconfgen_weights`.
- `baseUrl` — where bare names are downloaded from, e.g. a self-hosted mirror. Defaults
  to the Hugging Face repo. A mirror must serve the files uncompressed with
  `Content-Length` (`Content-Encoding` is not visible cross-origin, so a compressed body
  fails the size check).
- `cacheName` — Cache API cache name in the browser (default `"mlconfgen-weights"`).
- `onProgress` — called as download bytes arrive.

`finetuneCheckpointOnnx` is not resolved this way — pass a path or bytes, as before.

For direct control, use the manager on its own:

```js
import { WeightsManager } from "mlconfgen";

const wm = new WeightsManager({ cacheDir: "/shared/mlconfgen_weights" });
const path = await wm.resolve("egnn_chembl_15_39.onnx");
await wm.resolve("egnn_chembl_15_39.onnx", { forceDownload: true });
await wm.listAvailableWeights({ suffixes: [".onnx"] });
```

## Random molecules

`random()` generates molecules from a context picked at random from a bundled list of
ChEMBL molecule shapes, with the heavy-atom count of the picked molecule:

```js
const mols = await gen.random({ size: 5, seed: 7 });
```

`seed` only fixes which context is picked; the diffusion noise follows the global
`seed(n)`, so a fully reproducible run needs both. **Seeds are not compatible with the
Python package:** Python picks with `numpy.random.default_rng` (PCG64), this package with
a NumPy-compatible MT19937, so the same `seed` picks different contexts in the two.
Geometry optimisation (`optimize_geometry` in Python) is not available.

## Browser

To run in the browser (or any WebAssembly build), install `onnxruntime-web` and pass it
instead:

```js
import { createGenerator } from "mlconfgen";
import * as ort from "onnxruntime-web";

// Default weights, fetched from Hugging Face and cached in the browser
const gen = await createGenerator({ ort });
```

In the browser a bare name is fetched from `baseUrl` and cached with the Cache API
(HTTPS or localhost only; the browser may evict it). A bare name is not page-relative:
for weights served by your app, use `./egnn_chembl_15_39.onnx` or an absolute path.
Paths with a directory part (`./…`, `/…`) and URLs are passed to onnxruntime-web
unchanged:

```js
const gen = await createGenerator({
  ort,
  egnnOnnx: "/models/egnn_chembl_15_39.onnx",
  adjMatSeerOnnx: "/models/adj_mat_seer_chembl_15_39.onnx",
});

// …or bytes, e.g. files the user picked with an <input type="file">
const gen2 = await createGenerator({
  ort,
  egnnOnnx: new Uint8Array(await egnnFile.arrayBuffer()),
  adjMatSeerOnnx: new Uint8Array(await adjFile.arrayBuffer()),
});
```

`gen.clearCache()` deletes the Cache API cache.

RDKit is resolved automatically for the runtime it finds, so no extra wiring is needed
under a bundler (Vite, webpack, …), from a `<script>` tag that defines a global
`initRDKitModule`, or with no bundler at all. Pass `rdkitLoader` only to pin a specific
RDKit build:

```js
const gen = await createGenerator({
  ort,
  rdkitLoader: () => initRDKitModule({ locateFile: () => "/wasm/RDKit_minimal.wasm" }),
});
```

You can pass coordinates and let the package compute the context:

```js
const mols = await gen.generateConformers({
  referenceConformer: { positions: flatXyzFloat32 }, // length n*3
  nSamples: 10,
});
```

RDKit (bundled `@rdkit/rdkit`) is used automatically for validity filtering and SMILES
canonicalisation, in both Node and the browser. Pass a `rdkitLoader` function to
`createGenerator` to supply your own build.

If RDKit is configured but fails to initialise, generation throws an error named
`RdkitLoadError` rather than quietly treating every molecule as invalid and
returning an empty result set.

## Tests

```bash
cd js
npm test          # fast unit tests (offline)
npm run test:slow # ONNX generation (needs weights)
npm run test:all
```

The slow tests look for the weights in `js/`, the repository root and
`~/.mlconfgen_weights`, and run once per weight set found (core and distilled). Optional
env vars for the core model paths: `EGNN_ONNX`, `ADJ_ONNX`.

## Smoke UI

```bash
npm run smoke            # with RDKit (default)
npm run smoke:no-rdkit   # NO_RDKIT=1
# open http://localhost:3847
```

Optional env vars: `PORT`, `EGNN_ONNX`, `ADJ_ONNX`.


## Notes / limitations

- **Runtime** — bring your own ONNX Runtime (`onnxruntime-node` or `onnxruntime-web`), passed to `createGenerator` as `ort`. `@rdkit/rdkit` is bundled and loads in both Node and the browser. Node 18+.
- **RDKit.js** — SMILES atom-order canonicalisation before AdjMatSeer, plus validity filtering via sanitize.
- **Validity** — `generateConformers` defaults to `filterInvalid: true`.
- **RNG / float64** — same `seed(n)` as `np.random.seed(n)`.
- **Fine-tune adapter** — pass `finetuneCheckpointOnnx` for the optional EDM adapter.
- **`random()` seeds** — not cross-compatible with Python seeds (see [Random molecules](#random-molecules)).
- Fixed-fragment inpainting / IFM merge from the Python API are not ported.
