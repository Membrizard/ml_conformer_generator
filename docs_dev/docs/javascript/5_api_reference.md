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
  WeightsManager,
  ATOM_DECODER,
  CONTEXT_NORMS,
  DIMENSION,
  HF_REPO,
  MAX_N_NODES,
  MIN_N_NODES,
} from "mlconfgen";
```

---

## `createGenerator(options) → Promise<MLConformerGenerator>`

Resolves the weights through a [`WeightsManager`](#weightsmanager) (downloading them on first use), loads the ONNX sessions and returns a ready generator. See [Generation](2_generation.md#creategeneratoroptions) for all options.

Throws `TypeError` if `ort` is missing or does not expose `InferenceSession`, and an `Error` if a weights file cannot be resolved.

---

## `MLConformerGenerator`

Returned by `createGenerator`; can also be constructed directly from pre-created sessions:

```js
new MLConformerGenerator({ ort, generativeModel, adjMatSeer, edmAdapter = null, dimension, minNNodes, maxNNodes, contextNorms, atomDecoder, weightsManager = new WeightsManager() })
```

`weightsManager` is used by `listWeights` and `clearCache`.

### `generateConformers(options) → Promise<Molecule[]>`

Full pipeline: EDM → AdjMatSeer → optional standardisation and validity filter.

```js
{ referenceConformer, referenceContext, nAtoms, nSamples = 10, variance = 2,
  resampleSteps = 0, keepLargestFragment = true, filterInvalid = true }
```

### `random({ size = 1, seed = null, ...options }) → Promise<Molecule[]>`

Generates `size` samples from a context picked from the bundled list of ChEMBL molecule contexts, with `variance: 0`. `seed` (an integer in `[0, 2**32)`, or `null` for a random pick) fixes the pick, using its own generator, so the global `seed()` state is untouched. `resampleSteps`, `filterInvalid` and `keepLargestFragment` are passed to `generateConformers`. The seed list is loaded on the first call. Seeds are not compatible with the Python package; see [Generation](2_generation.md#randomoptions).

### `listWeights() → Promise<{ remote: string[], local: string[], remoteError?: Error }>`

`.onnx` weights on Hugging Face and in the cache (`WeightsManager.listAvailableWeights({ suffixes: [".onnx"] })`).

### `clearCache() → Promise<void>`

Deletes every cached weights file.

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

## `WeightsManager`

Resolves weights by name: existing path → cache → download. The JS counterpart of the Python `mlconfgen.utils.WeightsManager`.

```js
new WeightsManager({ cacheDir, baseUrl, cacheName = "mlconfgen-weights", onProgress, lockTimeout = 60000, staleGraceMs = 5000 })
```

| Option | Description |
|---|---|
| `cacheDir` | Cache directory (Node). Defaults to the `MLCONFGEN_CACHE` env var, then `~/.mlconfgen_weights`. |
| `baseUrl` | Download location for bare file names. Defaults to `https://huggingface.co/Membrizard/ml_conformer_generator/resolve/main`. A mirror's `Content-Length` enables the size check and the progress `total`; without it downloads still work. A cross-origin mirror must serve the files uncompressed or expose `Content-Encoding` (`Access-Control-Expose-Headers`): a hidden encoding makes the size check fail. |
| `cacheName` | Cache API cache name (browser). |
| `onProgress` | `({ file, loaded, total }) => void`, called as download bytes arrive. |
| `lockTimeout` | Milliseconds to wait for another process's download of the same file (Node). The wait restarts whenever that download makes progress; see [Locking](#locking). |
| `staleGraceMs` | Milliseconds the `<name>.part` of a lock whose process no longer exists must stay unchanged before the lock is reclaimed (Node); see [Locking](#locking). |

### `resolve(nameOrPath, { forceDownload = false }) → Promise<string | Uint8Array>`

Byte buffers are returned unchanged. In Node an existing file path is returned as is; a bare file name (`*.onnx`, no directory part) is returned as its cache path, downloaded first on a miss or with `forceDownload`; any other string throws. In the browser a bare file name resolves to its bytes (from the Cache API, or fetched and cached); any other string is returned unchanged for onnxruntime-web to load.

A download is written to `<name>.part` and renamed when complete. When the server sends `Content-Length`, a body of a different size fails the download. Before the rename, the download checks that `<name>.part` is still the file it created (same inode). If another process replaced or removed it, the download fails and leaves that file alone.

#### Locking

In Node a download holds `<stem>.lock` in the cache directory (e.g. `egnn_chembl_15_39.lock`), the same lock file the Python package uses:

- The lock file contains `<pid>@<hostname>` of the downloading process. When the download ends, the process removes the lock only if it still holds its own `<pid>@<hostname>`.
- A waiting process reclaims a lock only when both hold:
    - the lock's PID no longer exists on this host. A lock holding only a PID counts as same-host;
    - the holder's `<name>.part` is absent or has not changed for `staleGraceMs` (5 s). This keeps a live download from being taken over when its PID is not visible, e.g. from a container with its own PID namespace but the same hostname.
- After Ctrl-C during a download, the next process therefore takes over about 5 s after the last write to `<name>.part`.
- Locks from other hosts, and the Python package's empty locks, are waited on.
- The wait times out after `lockTimeout` ms without progress. Each change to the holder's `<name>.part` restarts the timer, so a long download by another JS process does not time out its waiters. The Python package does not write `<name>.part`, so a JS process waiting on a Python download times out after `lockTimeout`.

Known gaps:

- Two processes that reclaim the same stale lock at the same moment may both download; one of them may fail, and a rerun recovers.
- Releasing reads the lock and then deletes it. A takeover that lands between the two loses its lock, so a third process may start another download.

The `.part` check above bounds the effect of both. A download whose `<name>.part` was replaced fails instead of putting another process's unfinished file into the cache, and a rerun recovers. Only a takeover in the instant between that check and the rename gets past it. Both gaps need a holder that crashed or whose PID is not visible, plus a window a few system calls wide.

### `listAvailableWeights({ suffixes = [".pt", ".onnx"] }) → Promise<{ remote, local, remoteError? }>`

`remote` lists the Hugging Face repo (regardless of `baseUrl`); `local` lists the cache directory (relative paths) in Node and the Cache API cache (file names) in the browser. A failed Hugging Face request sets `remoteError` and leaves `remote` empty.

### `clearCache() → Promise<void>`

Removes the cache directory's contents (Node) or deletes the Cache API cache (browser). In Node it refuses to clear a cache directory that is the filesystem root, the home or working directory, or a parent of either. Symlinks are resolved first.

### `getCacheDir() → Promise<string>`

Absolute cache directory (Node only).

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
| `HF_REPO` | `"Membrizard/ml_conformer_generator"` — Hugging Face repo hosting the weights |

---

## Errors

| Error | Raised when |
|---|---|
| `TypeError` | `ort` missing/invalid; invalid `predictBonds` value; `filterInvalid` with `predictBonds: "never"`. |
| `RangeError` | Empty heavy-atom range after clamping; malformed `Molecule` positions; V2000 overflow; `random` seed outside `[0, 2**32)`. |
| `RdkitLoadError` | RDKit loader configured but failed to initialise (`error.name === "RdkitLoadError"`). |
| `Error` | Neither reference form supplied to `prepareInputs`; weights file not found (`Weights file not found: …`); download failed, in Node or the browser (`Could not resolve path to the weights file: <name>: <cause>`, with `error.cause`; the cause may be an HTTP error, `Size mismatch for <name>: …` or `Another process replaced or removed <name>.part …`); `clearCache()` on the filesystem root, the home or working directory, or a parent of either (`Refusing to clear …`). |
| `TimeoutError` | Another process held the download lock for `lockTimeout` ms without progress on its download (`error.name === "TimeoutError"`). |
