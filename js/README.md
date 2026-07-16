# ML Conformer Generator JS (JavaScript / ONNX Runtime)

Node.js package for spatially-aware molecule generation. Ships with **onnxruntime-node** and **@rdkit/rdkit**

## Install

```bash
npm install mlconfgen
```

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
import { MLConformerGenerator, seed } from "mlconfgen";

seed(42);

const gen = await MLConformerGenerator.create({
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

`create()` also works as `import { create } from "mlconfgen"`.

You can pass coordinates and let the package compute the context:

```js
const mols = await gen.generateConformers({
  referenceConformer: { positions: flatXyzFloat32 }, // length n*3
  nSamples: 10,
});
```

To use a different ONNX Runtime build, pass `ort` explicitly (e.g. `onnxruntime-web`). To skip RDKit sanitize / SMILES reorder, call `clearRdkitLoader()` before generating.

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

- **Dependencies** — `onnxruntime-node` + `@rdkit/rdkit` (installed with the package). Node 18+.
- **RDKit.js** — SMILES atom-order canonicalisation before AdjMatSeer, plus validity filtering via sanitize.
- **Validity** — `generateConformers` defaults to `filterInvalid: true` 
- **RNG / float64** —  Same `seed(n)` as `np.random.seed(n)`.
- **Fine-tune adapter** — pass `finetuneCheckpointOnnx` for the optional EDM adapter.
- Fixed-fragment inpainting / IFM merge from the Python API are not ported.
