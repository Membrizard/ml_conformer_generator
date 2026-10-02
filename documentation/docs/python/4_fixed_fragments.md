# Fixed Fragments

Fixed-fragment generation (inpainting) keeps a chosen substructure in place and lets the EDM generate the remaining atoms around it. It supports scaffold hopping, fragment growing and R-group exploration while preserving a binding motif.

---

## Overview

During denoising the known fragment coordinates and atom types are blended into the trajectory at every step, following a polynomial schedule `(1 − t)^blend_power`. The generated atoms adapt to the fixed ones; the fixed ones stay (up to numerical noise) where you put them.

Two ways to define the fragment are supported:

- A **set of atom indices** of the reference conformer
- A **separate `Mol` object** with a 3D conformer

---

## 1. Fragment as Atom Indices

The simplest option when the fragment is part of the reference molecule:

```python
from rdkit import Chem
from mlconfgen import MLConformerGenerator

model = MLConformerGenerator(diffusion_steps=100)
reference = Chem.MolFromMolFile("./assets/demo_files/ceyyag.mol")

samples = model.generate_conformers(
    reference_conformer=reference,
    fixed_fragment={3, 5, 6, 7, 8, 9, 10},   # heavy-atom indices of the reference
    n_samples=20,
    variance=1,
    resample_steps=4,
    blend_power=3,
)
```

Indices refer to the heavy atoms of the reference **after hydrogens are removed** (`Chem.RemoveAllHs`). The fragment is extracted and aligned together with the reference, so no coordinate handling is required.

---

## 2. Fragment as a Molecule

Use a `Mol` when the fragment comes from a different source (e.g. a docked fragment hit):

```python
from mlconfgen.utils import extract_fragment

fragment = extract_fragment(reference, {3, 5, 6, 7, 8, 9, 10})   # or Chem.MolFromMolFile(...)

samples = model.generate_conformers(
    reference_conformer=reference,
    fixed_fragment=fragment,
    n_samples=20,
    variance=1,
    resample_steps=4,
)
```

### Coordinate frame rules

- With a `reference_conformer`, the fragment must be defined **in the same coordinate system as the reference**. The reference transform (centering + rotation to the principal frame) is applied to the fragment automatically.
- With a `reference_context`, there is no reference transform. The fragment must already be expressed **in the principal inertial frame of the target shape**, and only a `Mol` is accepted:

```python
samples = model.generate_conformers(
    reference_context=context,
    n_atoms=17,
    fixed_fragment=fragment_in_principal_frame,
    n_samples=20,
    variance=1,
    resample_steps=4,
)
```

Passing a `set` together with a `reference_context` raises `ValueError`.

---

## 3. Recommendations

- Fragments of **≤ 14–17 heavy atoms** give the best results; keep **≥ 12–15 atoms free** so the model has room to generate.
- Use `resample_steps=4–10` — harmonisation matters much more with a fixed fragment than for free generation.
- Keep `blend_power=3`. Lower values enforce the fragment more rigidly at the cost of worse integration with the generated part; `0` is a hard injection.
- Validity is lower than for free generation (expect ≥ 10–30% depending on the fragment). Increase `n_samples` accordingly.
- The `edm_moi_chembl_15_39_inpaint.pt` weights are an alternative EDM tuned for inpainting workflows.
- For higher shape similarity and more stable fragment integration consider [Inertial Fragment Matching with a fixed fragment](5_inertial_fragment_matching.md#2-fixed-fragment-ifm).

---

## 4. Utilities

```python
from mlconfgen.utils import extract_fragment, align_mol_to_principal_frame, set_conformer_positions

# Sub-molecule from heavy-atom indices (keeps 3D coordinates)
frag = extract_fragment(mol, {0, 1, 2, 3})

# Reference in its principal inertial frame + the transform used
context, shift, rotation, aligned_coord = align_mol_to_principal_frame(mol)
aligned_mol = set_conformer_positions(Chem.Mol(mol), aligned_coord)
```
