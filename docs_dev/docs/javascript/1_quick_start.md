# Quick Start

The `mlconfgen` npm package runs the ONNX weights with ONNX Runtime — no Python or PyTorch required. It mirrors the main generation path of `MLConformerGeneratorONNX`: context → EDM samples → AdjMatSeer bonds → standardisation and validity filtering.

---

## Install

```bash
npm install mlconfgen onnxruntime-node
```

The package bundles `@rdkit/rdkit` (used for canonical atom ordering and sanitisation) and expects you to supply the ONNX Runtime. Node 18+.

Download the ONNX weights from Hugging Face and keep them next to your application (see [Model Weights](../getting_started/2_model_weights.md)):

- `egnn_chembl_15_39.onnx`
- `adj_mat_seer_chembl_15_39.onnx`

---

## Generate

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
  referenceContext: [89.87, 210.78, 217.78], // principal MOI components
  nAtoms: 20,
  nSamples: 10,
  variance: 2,
});

for (const mol of mols) {
  console.log(`atoms=${mol.nAtoms} bonds=${mol.bonds.length}`);
  console.log(mol.toMolBlock());
}
```

`createGenerator(options)` loads both ONNX sessions and returns an `MLConformerGenerator`. `generateConformers` returns an array of `Molecule` objects that passed validity filtering.

---

## Reference from Coordinates

Instead of supplying the MOI context, pass heavy-atom coordinates and let the package compute it:

```js
const mols = await gen.generateConformers({
  referenceConformer: { positions: flatXyzFloat32 }, // Float32Array of length n*3
  nSamples: 10,
});
```

`positions` may be a flat array, a `Float32Array`, or an array of `[x, y, z]` triples. `nAtoms` defaults to the number of points.

---

## Reproducibility

`seed(n)` seeds a NumPy-compatible `RandomState` (MT19937) used for all noise draws. With the same seed and weights, `randn` draws match `np.random.seed(n)` in Python, so runs are comparable across the two implementations. Unseeded runs are initialised from crypto entropy.

---

## Run the Examples

From a repository checkout:

```bash
cd js
npm install
npm run example                 # examples/generate.js
STEPS=50 node examples/animate.js   # writes trajectory.xyz
```

Both examples read `EGNN_ONNX` / `ADJ_ONNX` environment variables for weight paths.

### Tests and smoke UI

```bash
npm test          # fast unit tests
npm run test:slow # ONNX generation (needs weights)
npm run test:all

npm run smoke     # local trajectory viewer at http://localhost:3847
```

---

## Next

- [Generation](2_generation.md) — all options and the `Molecule` object
- [Browser Usage](3_browser.md) — `onnxruntime-web`, loading weights client-side
- [Animating Generation](4_animation.md) — per-step trajectories
