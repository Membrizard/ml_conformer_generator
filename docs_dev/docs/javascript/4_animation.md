# Animating Generation

`animateGeneration` is a streaming variant of `generateConformers`. It yields one frame per denoising step, each frame being the whole batch decoded to `Molecule` objects at that step — ideal for visualising the diffusion trajectory.

---

## Usage

```js
import { createGenerator, seed } from "mlconfgen";
import * as ort from "onnxruntime-node";

const gen = await createGenerator({
  ort,
  egnnOnnx: "./egnn_chembl_15_39.onnx",
  adjMatSeerOnnx: "./adj_mat_seer_chembl_15_39.onnx",
  diffusionSteps: 50,
});

seed(42);

for await (const { step, total, molecules } of gen.animateGeneration({
  referenceContext: [89.87, 210.78, 217.78],
  nAtoms: 20,
  nSamples: 1,
  variance: 0,
})) {
  const mol = molecules[0];
  console.log(`step ${step}/${total}: ${mol.nAtoms} atoms, ${mol.bonds.length} bonds`);
}
```

Each yielded frame is `{ step, total, molecules }` with `step` running from `1` to `total` (= `diffusionSteps`).

---

## Options

All `generateConformers` reference options apply (`referenceConformer` / `referenceContext` + `nAtoms`, `nSamples`, `variance`, `resampleSteps`), plus:

| Option | Type | Default | Description |
|---|---|---|---|
| `predictBonds` | `"last"` / `"always"` / `"never"` | `"last"` | When to run AdjMatSeer. |
| `filterInvalid` | bool | `false` | Standardise and validity-filter the **final** frame only. |
| `keepLargestFragment` | bool | `true` | Only meaningful with `filterInvalid`. |

### `predictBonds`

- `"last"` — AdjMatSeer on the final frame only. Intermediate frames are atom clouds; the whole animation costs one extra inference.
- `"always"` — AdjMatSeer on every frame, so a viewer can show connectivity forming instead of guessing bonds by distance. Costs one AdjMatSeer call per step.
- `"never"` — atom clouds only.

### `filterInvalid`

By default the final frame carries bonds but is **not** passed through standardisation, so it may be multi-fragment or fail sanitisation. With `filterInvalid: true` the last frame is processed exactly like a `generateConformers` result — which can drop molecules, so the final frame may be shorter than the preceding ones (possibly empty). Requires `predictBonds` of `"last"` or `"always"`.

> **Note:** Decoding every step consumes RNG, so the final animated frame is a valid sample but not bit-identical to a single `generateConformers` call under the same seed. The EDM fine-tune adapter is not applied per frame.

---

## Example: Write an XYZ Trajectory

`js/examples/animate.js` collects the frames into a multi-frame `.xyz` file that can be played in MolView, VMD, PyMOL or Avogadro:

```js
import { writeFileSync } from "node:fs";

const Z = { 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl", 35: "Br" };

const frames = [];
for await (const { step, total, molecules } of gen.animateGeneration({
  referenceContext: [89.87, 210.78, 217.78],
  nAtoms: 20,
  nSamples: 1,
  variance: 0,
})) {
  const m = molecules[0];
  const lines = [String(m.nAtoms), `step ${step}/${total}`];
  for (let i = 0; i < m.nAtoms; i += 1) {
    const [x, y, z] = m.positions.subarray(i * 3, i * 3 + 3);
    lines.push(`${Z[m.atomicNumbers[i]]} ${x.toFixed(4)} ${y.toFixed(4)} ${z.toFixed(4)}`);
  }
  frames.push(lines.join("\n"));
}

writeFileSync("trajectory.xyz", `${frames.join("\n")}\n`);
```

Run the bundled version with `STEPS=50 node examples/animate.js` from the `js` directory.

---

## Lower Level: `EquivariantDiffusion.animate`

For custom tensors, the diffusion model itself exposes the same generator:

```js
import { EquivariantDiffusion } from "mlconfgen";

const edm = await EquivariantDiffusion.create("./egnn_chembl_15_39.onnx", ort, { timesteps: 50 });
for await (const { x, h, step, total } of edm.animate(nodeMask, edgeMask, context)) {
  // x: { data: Float32Array, shape: [B, N, 3] }, h: { data, shape: [B, N, 8] }
}
```

`edm.sample(nodeMask, edgeMask, context, resampleSteps)` is the non-streaming equivalent.
