# Generation
*(Detailed Guide)*

This document covers `createGenerator` options, `generateConformers` parameters, the returned `Molecule` objects, and the lower-level pipeline methods.

---

## `createGenerator(options)`

```js
const gen = await createGenerator({
  ort,                                        // required
  egnnOnnx: "./egnn_chembl_15_39.onnx",
  adjMatSeerOnnx: "./adj_mat_seer_chembl_15_39.onnx",
  finetuneCheckpointOnnx: null,
  diffusionSteps: 100,
  rdkitLoader: undefined,
});
```

| Option | Type | Default | Description |
|---|---|---|---|
| `ort` | object | — | ONNX Runtime namespace: `import * as ort from "onnxruntime-node"` or an `onnxruntime-web` build. ✅ required |
| `egnnOnnx` | string / `Uint8Array` | `./egnn_chembl_15_39.onnx` | EGNN denoiser model — file path (Node) or model bytes. |
| `adjMatSeerOnnx` | string / `Uint8Array` | `./adj_mat_seer_chembl_15_39.onnx` | AdjMatSeer model. |
| `finetuneCheckpointOnnx` | string / `Uint8Array` / null | `null` | Optional EDM adapter exported from an RL fine-tuning checkpoint. |
| `diffusionSteps` | int | `100` | Number of denoising steps. |
| `rdkitLoader` | `() => Promise<RDKitModule>` | bundled | Custom RDKit.js loader; see [Browser Usage](3_browser.md). |
| `minNNodes` / `maxNNodes` | int | `15` / `39` | Hard clamp on heavy-atom counts. |
| `dimension`, `contextNorms`, `atomDecoder` | | | Model constants; override only for custom weights. |

Distilled weights are loaded the same way:

```js
const gen = await createGenerator({
  ort,
  egnnOnnx: "./small_egnn_chembl_15_39.onnx",
  adjMatSeerOnnx: "./small_adj_mat_seer_obabel_15_39.onnx",
});
```

---

## `generateConformers(options)`

```js
const mols = await gen.generateConformers({
  referenceConformer: null,
  referenceContext: null,
  nAtoms: null,
  nSamples: 10,
  variance: 2,
  resampleSteps: 0,
  keepLargestFragment: true,
  filterInvalid: true,
});
```

| Parameter | Type | Default | Required | Description |
|---|---|---|---|---|
| `referenceConformer` | `{ positions }` / `Float32Array` / number[] | `null` | ✅ one of | Heavy-atom coordinates; the MOI context is computed internally. |
| `referenceContext` | number[3] | `null` | ✅ one of | Principal MOI components. |
| `nAtoms` | int | `null` | ⚠️ with context | Target heavy-atom count. Defaults to the conformer size when coordinates are given. |
| `nSamples` | int | `10` | | Number of samples to draw. |
| `variance` | int | `2` | | Allowed ± deviation in heavy-atom count. |
| `resampleSteps` | int | `0` | | Extra resampling per denoising step. |
| `keepLargestFragment` | bool | `true` | | Keep the largest connected component, or reject disconnected samples. |
| `filterInvalid` | bool | `true` | | Run standardisation + RDKit sanitisation and drop failures. |

Returns `Promise<Molecule[]>`.

### Detailed explanations

#### `referenceConformer` / `referenceContext` / `nAtoms`

Exactly one of the two references must be given. With `referenceContext`, `nAtoms` is mandatory because the shape carries no size information. Contexts are the eigenvalues of the moment-of-inertia tensor for equal-mass points, in ascending order — the same values the Python API uses, so contexts can be shared between the two libraries.

```js
import { contextFromCoordinates } from "mlconfgen";
const { context, aligned } = contextFromCoordinates(flatXyz, nAtoms);
```

#### `variance`

Sample sizes are drawn uniformly from `[nAtoms − variance, nAtoms + variance]` and clamped to `[minNNodes, maxNNodes]`. A `RangeError` is thrown if the range is empty after clamping.

#### `resampleSteps`

Additional forward/backward resampling per denoising step. Improves sample quality at a proportional cost in runtime; `0` is adequate for free generation.

#### `keepLargestFragment` and `filterInvalid`

With `filterInvalid: true` (default) each sample goes through `standardizeMol`: largest-fragment selection (or rejection of disconnected samples when `keepLargestFragment: false`), followed by RDKit sanitisation. Bond orders and aromaticity are refreshed from RDKit's output. With `filterInvalid: false` the raw bonded molecules are returned (largest fragment only, if requested) and nothing is dropped.

> **Note:** MMFF94 geometry optimisation from the Python pipeline is not available in RDKit.js; returned coordinates are the raw diffusion output.

---

## The `Molecule` Object

Lightweight container for heavy atoms, coordinates and bonds.

| Member | Description |
|---|---|
| `atomicNumbers` | `Int32Array` of atomic numbers. |
| `positions` | `Float32Array`, flat `[x0, y0, z0, x1, ...]` in Å. |
| `bonds` | `{ i, j, type }[]`, `type` ∈ `1` single, `2` double, `3` triple, `4` aromatic. |
| `nAtoms` | Number of atoms. |
| `fragmentCount()` | Number of connected components. |
| `largestFragment()` | New `Molecule` with only the largest component. |
| `toMolBlock(name?)` | V2000 mol block (also `toMolBlockV2000`). |
| `toMolBlockV3000()` | V3000 mol block. |

Mol blocks can be handed to RDKit.js (`RDKit.get_mol(block)`), 3Dmol.js, or written to `.mol`/`.sdf` files.

```js
import { Molecule } from "mlconfgen";
const mol = new Molecule({ atomicNumbers: [6, 8], positions: [0, 0, 0, 1.2, 0, 0], bonds: [{ i: 0, j: 1, type: 2 }] });
```

---

## Lower-Level Methods

The pipeline stages are exposed individually on the generator:

```js
const { referenceContext, nAtoms } = gen.prepareInputs({ referenceConformer: { positions } });

// Stage 1: atom clouds (no bonds)
const clouds = await gen.edmSamples({
  referenceContext,
  nSamples: 10,
  minNNodes: nAtoms - 2,
  maxNNodes: nAtoms + 2,
  resampleSteps: 0,
});

// Stage 2: bonds from AdjMatSeer (atoms are canonically reordered first)
const bonded = await gen.predictBonds(clouds);

// Stage 3: standardise yourself
import { standardizeMol, isValidMol } from "mlconfgen";
const valid = (await Promise.all(bonded.map((m) => standardizeMol(m)))).filter(Boolean);
```

`isValidMol(mol)` returns `1` / `0` using RDKit sanitisation (or `1` when no RDKit loader is configured).

---

## Fine-Tuning Checkpoints

An RL fine-tuning checkpoint exported with `export_to_onnx` ([Python docs](../python/8_onnx.md#exporting-a-fine-tuning-checkpoint)) can be applied as an EDM adapter:

```js
const gen = await createGenerator({
  ort,
  egnnOnnx: "./egnn_chembl_15_39.onnx",
  adjMatSeerOnnx: "./adj_mat_seer_chembl_15_39.onnx",
  finetuneCheckpointOnnx: "./finetune_checkpoint.onnx",
});
```

The adapter is applied after sampling in `generateConformers` / `edmSamples`. Use the AdjMatSeer ONNX exported from the same fine-tuned model to also pick up the fine-tuned bond predictor.

---

## Limitations

- Fixed-fragment inpainting and Inertial Fragment Matching are not ported; use the Python library for those.
- No MMFF94 geometry optimisation.
- `MIN_N_NODES` is `15` in the JS package (the Python default is `6` to accommodate the fragment model).
