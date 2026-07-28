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

Place the ONNX weights next to your app (or pass absolute paths). **Weights are not published on npm** (CC BY-NC-ND); download / export them separately:

- `egnn_chembl_15_39.onnx`
- `adj_mat_seer_chembl_15_39.onnx`

## Quick start

```js
import { createGenerator, seed } from "mlconfgen";
import * as ort from "onnxruntime-node";

seed(42);

const gen = await createGenerator({
  ort,
  egnnOnnx: "./egnn_chembl_15_39.onnx",
  adjMatSeerOnnx: "./adj_mat_seer_chembl_15_39.onnx",
  diffusionSteps: 100,
});

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

### Browser

To run in the browser (or any WebAssembly build), install `onnxruntime-web` and pass it
instead:

```js
import { createGenerator } from "mlconfgen";
import * as ort from "onnxruntime-web";

// Weights loaded from the user's own machine (e.g. an <input type="file">),
// so they are never uploaded or re-hosted.
const gen = await createGenerator({
  ort,
  egnnOnnx: new Uint8Array(await egnnFile.arrayBuffer()),
  adjMatSeerOnnx: new Uint8Array(await adjFile.arrayBuffer()),
});
```

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

The weights are **not** published on npm and are licensed CC BY-NC-ND — download them
from Hugging Face and keep them local to your machine or application. Point
`egnnOnnx` / `adjMatSeerOnnx` at those local files, as in the example above.

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
npm test          # fast unit tests
npm run test:slow # ONNX generation (needs weights)
npm run test:all
```

Optional env vars for model paths: `EGNN_ONNX`, `ADJ_ONNX`.

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
- Fixed-fragment inpainting / IFM merge from the Python API are not ported.
