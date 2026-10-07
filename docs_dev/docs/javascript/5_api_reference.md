# API Reference

Everything exported from the `mlconfgen` npm package (`import { ... } from "mlconfgen"`).

```js
import {
  createGenerator,
  EquivariantDiffusion,
  Molecule,
  contextFromCoordinates,
  isValidMol,
  standardizeMol,
  seed,
  NumpyRandomState,
  ATOM_DECODER,
  CONTEXT_NORMS,
  DIMENSION,
  MAX_N_NODES,
  MIN_N_NODES,
} from "mlconfgen";
```

---

## `createGenerator(options) → Promise<MLConformerGenerator>`

Loads the ONNX sessions and returns a ready generator. See [Generation](2_generation.md#creategeneratoroptions) for all options.

Throws `TypeError` if `ort` is missing or does not expose `InferenceSession`.

---

## `MLConformerGenerator`

Returned by `createGenerator`; can also be constructed directly from pre-created sessions:

```js
new MLConformerGenerator({ ort, generativeModel, adjMatSeer, edmAdapter = null, dimension, minNNodes, maxNNodes, contextNorms, atomDecoder })
```

### `generateConformers(options) → Promise<Molecule[]>`

Full pipeline: EDM → AdjMatSeer → optional standardisation and validity filter.

```js
{ referenceConformer, referenceContext, nAtoms, nSamples = 10, variance = 2,
  resampleSteps = 0, keepLargestFragment = true, filterInvalid = true }
```

### `animateGeneration(options) → AsyncGenerator<{ step, total, molecules }>`

Streaming variant yielding one frame per denoising step.

```js
{ referenceConformer, referenceContext, nAtoms, nSamples = 1, variance = 2, resampleSteps = 0,
  predictBonds = "last" | "always" | "never", filterInvalid = false, keepLargestFragment = true }
```

### `prepareInputs({ referenceConformer, referenceContext, nAtoms }) → { referenceContext: Float32Array, nAtoms }`

Resolves the reference into a context and a target size. Throws if neither a conformer nor context + `nAtoms` is provided.

### `edmSamples({ referenceContext, nSamples = 100, maxNNodes = 32, minNNodes = 25, resampleSteps = 0 }) → Promise<Molecule[]>`

Diffusion stage only; returns bond-less molecules. Applies the EDM adapter when configured. Throws `RangeError` if the node range is empty after clamping.

### `predictBonds(molecules) → Promise<Molecule[]>`

Canonicalises atom order (RDKit SMILES output order), runs AdjMatSeer and assigns bonds. Where AdjMatSeer predicts no bond but distance-based connectivity found one, a single bond is kept.

### `applyEdmAdapter(x, h, nodeMask, edgeMask) → Promise<{ x, h }>`

Runs the optional fine-tune adapter on raw tensors.

---

## `Molecule`

```js
new Molecule({ atomicNumbers, positions, bonds = [] })
```

| Member | Type |
|---|---|
| `atomicNumbers` | `Int32Array` |
| `positions` | `Float32Array` (flat, `nAtoms * 3`) |
| `bonds` | `{ i: number, j: number, type: 1\|2\|3\|4 }[]` |
| `nAtoms` | `number` |
| `fragmentCount()` | `number` |
| `largestFragment()` | `Molecule` |
| `toMolBlock(name = "MLConfGen")` / `toMolBlockV2000(name)` | `string` (V2000) |
| `toMolBlockV3000()` | `string` (V3000) |

Throws `RangeError` if `positions.length !== atomicNumbers.length * 3`, or when exporting more than 999 atoms/bonds as V2000.

---

## Molecule Utilities

### `contextFromCoordinates(positions, nAtoms = positions.length / 3) → { context: Float32Array(3), aligned: Float32Array }`

Centres the point cloud, diagonalises its moment-of-inertia tensor and returns the ascending principal moments plus the coordinates rotated into the principal frame. Equal masses are assumed, matching the Python pipeline.

### `isValidMol(mol) → Promise<0 | 1>`

RDKit sanitisation check. Returns `1` when no RDKit loader is configured. Throws `RdkitLoadError` if RDKit is configured but fails to initialise.

### `standardizeMol(mol, { keepLargestFragment = true }) → Promise<Molecule | null>`

Largest fragment (or `null` for disconnected molecules when `keepLargestFragment` is `false`), then RDKit sanitisation; bond orders are refreshed from RDKit. Returns `null` on failure.

---

## Random Number Generation

### `seed(value = null)`

Seeds the shared NumPy-compatible `RandomState` (MT19937 + randomkit) used for all noise draws. `null` reseeds from entropy. Behaves like `np.random.seed(value)` for integers in `[0, 2**32)`.

### `NumpyRandomState`

The underlying generator class, exposed for advanced use (`new NumpyRandomState(seed)`). Its `randn` draws match NumPy's legacy generator bit-for-bit in float64.

---

## `EquivariantDiffusion`

The diffusion sampler wrapping the EGNN ONNX session.

```js
const edm = await EquivariantDiffusion.create(egnnOnnx, ort, { timesteps = 100, noisePrecision = 1e-5 });
await edm.sample(nodeMask, edgeMask, context, resampleSteps = 0);      // → { x, h }
for await (const f of edm.animate(nodeMask, edgeMask, context, 0)) {} // → { x, h, step, total }
```

Tensors are plain `{ data: Float32Array, shape: number[] }` objects. `nodeMask` has shape `[B, N, 1]`, `edgeMask` `[B*N*N, 1]`, `context` `[B, N, 3]` (normalised MOI broadcast over nodes).

---

## Constants

| Name | Value |
|---|---|
| `DIMENSION` | `42` — padded size of AdjMatSeer inputs |
| `MIN_N_NODES` / `MAX_N_NODES` | `15` / `39` |
| `ATOM_DECODER` | `{ 0: "C", 1: "N", 2: "O", 3: "F", 4: "P", 5: "S", 6: "Cl", 7: "Br" }` |
| `CONTEXT_NORMS` | `{ mean: [105.0766, 473.1938, 537.4675], mad: [52.0409, 219.7475, 232.9718] }` |

---

## Errors

| Error | Raised when |
|---|---|
| `TypeError` | `ort` missing/invalid; invalid `predictBonds` value; `filterInvalid` with `predictBonds: "never"`. |
| `RangeError` | Empty heavy-atom range after clamping; malformed `Molecule` positions; V2000 overflow. |
| `RdkitLoadError` | RDKit loader configured but failed to initialise (`error.name === "RdkitLoadError"`). |
| `Error` | Neither reference form supplied to `prepareInputs`. |
