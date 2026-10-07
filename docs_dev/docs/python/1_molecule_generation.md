# Molecule Generation

This page describes basic usage of the `mlconfgen` Python library to generate molecules with the PyTorch backend. The ONNX backend exposes the same interface — see [ONNX Inference & Export](8_onnx.md).

Interactive examples are available in the repository: `python_api_demo.ipynb`.

---

## Overview

Generation is a three-stage pipeline wrapped by a single call:

- **EDM sampling**: the Equivariant Diffusion Model samples heavy-atom coordinates and element types conditioned on the reference shape.
- **Bond prediction**: the AdjMatSeer GCN predicts the adjacency matrix (bond orders) from the atom cloud.
- **Standardisation**: largest-fragment selection, valence check, kekulisation, RDKit sanitisation and optional constrained MMFF94 geometry optimisation. Molecules that fail are dropped.

The result is a list of **valid** RDKit `Mol` objects. The number of returned molecules is therefore lower than `n_samples`.

---

## 1. Generation from a Reference Conformer

```python
from rdkit import Chem
from mlconfgen import MLConformerGenerator, evaluate_samples

model = MLConformerGenerator(
    edm_weights="edm_moi_chembl_15_39.pt",
    adj_mat_seer_weights="adj_mat_seer_chembl_15_39.pt",
    diffusion_steps=100,
)

reference = Chem.MolFromMolFile("./assets/demo_files/ceyyag.mol")

samples = model.generate_conformers(
    reference_conformer=reference,
    n_samples=20,
    variance=2,
)

print(f"{len(samples)} valid molecules")
```

The reference must carry a 3D conformer. Hydrogens are stripped automatically; the reference heavy-atom count sets the target size of the generated molecules.

> **Note:** The default weights are trained on molecules with **15–39 heavy atoms**. References outside this range still work but generation quality degrades.

---

## 2. Generation from an Arbitrary Context

Instead of a molecule, pass the three principal moments of inertia directly along with the target number of heavy atoms:

```python
import torch

context = torch.tensor([89.87, 210.78, 217.78])   # principal MOI components

samples = model.generate_conformers(
    reference_context=context,
    n_atoms=20,
    n_samples=20,
)
```

This is how shape-only references (e.g. a pocket volume) are supplied. The context can be computed for any point cloud with `align_mol_to_principal_frame` or `get_moment_of_inertia_tensor` from `mlconfgen.utils`.

---

## 3. Random Generation

When no reference is at hand — smoke tests, unconditional sampling, seeding a library — `random()` draws a context from a bundled set of **ChEMBL-derived seeds** (`random_molecule_seeds.json`, ~100 contexts per heavy-atom count from 15 to 39, selected to reproduce the MOI distribution of the training set) and generates molecules for it:

```python
mols = model.random(size=10, seed=42)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `size` | int | `1` | Number of molecules to generate. |
| `seed` | int / None | `None` | Seed for the context draw; the same seed picks the same context. |
| `optimize_geometry` | bool | `True` | MMFF94 optimisation during standardisation. |

The picked seed fixes both the shape (`context`) and the heavy-atom count (`n_atoms`); generation runs with `variance=0`, so every molecule in the batch has the same size. The method is available on both backends with the same signature:

```python
from mlconfgen import MLConformerGenerator, MLConformerGeneratorONNX

MLConformerGenerator().random(size=5)
MLConformerGeneratorONNX().random(size=5, seed=7)
```

`seed` only controls which context is picked. To make the diffusion itself reproducible, seed the backend RNG as well (`torch.manual_seed` for PyTorch, `np.random.seed` for ONNX). The underlying helper is exposed as `mlconfgen.utils.random_context(size, seed)` and returns `{"context": [I1, I2, I3], "n_atoms": n}` for use with `generate_conformers` directly.

---

## 4. Evaluating Samples

`evaluate_samples` aligns every sample to the reference and scores shape and chemical similarity:

```python
aligned_reference, results = evaluate_samples(reference, samples)

for item in results:
    print(item["shape_tanimoto"], item["chemical_tanimoto"])
```

- `aligned_reference`: mol block of the reference in its principal inertial frame
- `results`: one dict per sample with
    - `mol_block` — the sample aligned to the reference frame
    - `shape_tanimoto` — Shape Tanimoto similarity via Gaussian volume overlap (hydrogens ignored)
    - `chemical_tanimoto` — Tanimoto similarity of 2-hop 2048-bit Morgan fingerprints

Pass `sanitize_ref=False` when the reference cannot be sanitised (e.g. a pocket pseudo-molecule); chemical similarity is then reported as `0`.

How alignment and the shape score are computed is described in [Shape Similarity & Alignment](6_shape_similarity.md).

---

## 5. Lower-Level Access

The pipeline stages are exposed individually for custom workflows:

```python
# Prepared context, target size and (optionally) prepared fixed fragment
ref_context, ref_n_atoms, _ = model.prepare_inputs(reference_conformer=reference)

# Stage 1: atom clouds without bonds
edm_mols = model.edm_samples(
    reference_context=ref_context,
    n_samples=20,
    min_n_nodes=ref_n_atoms - 2,
    max_n_nodes=ref_n_atoms + 2,
)

# Stage 2: bonds from AdjMatSeer
raw_mols = model.predict_bonds(edm_mols)

# Stage 3: standardise yourself
from mlconfgen.utils import standardize_mol
valid = [m for m in (standardize_mol(x, optimize_geometry=True) for x in raw_mols) if m]
```

`edm_samples(..., raw_output=True)` returns the raw `(x, h, node_mask, edge_mask)` tensors instead of molecules.

Stage 2 can be swapped for a deterministic, rule-based bond perception — see [Bond Prediction](3_bond_prediction.md).

---

## 6. Practical Tips

- Generation is batched; request `n_samples` in the tens to hundreds rather than looping over single samples.
- `diffusion_steps=100` is the quality/speed sweet spot. Values of 20–50 are usable for screening; avoid values below 20.
- Expect roughly 50% of samples to survive standardisation with the core AdjMatSeer and ~70% with the distilled one.
- Call `model.generate_conformers` under `torch.inference_mode()` if you wrap it in your own loop — the method already does so internally.
