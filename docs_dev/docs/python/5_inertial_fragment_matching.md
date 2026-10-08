# Inertial Fragment Matching

**Inertial Fragment Matching (IFM)** is an MLConfGen-specific inference strategy that exploits the **additivity of the Moment of Inertia tensor**. Instead of generating a whole molecule in one pass, the reference shape is decomposed into fragments, shape-matched fragments are generated independently, placed at their predicted positions and merged with a short denoising pass.

In benchmarks IFM raises average Shape Tanimoto similarity from ~53% (basic generation) to ~70%.

---

## Models

IFM uses two generators:

- **`generator`** — produces fragments. Use the fragment EDM `edm_moi_chembl_6_39_fragments.pt`, which is trained down to 6 heavy atoms.
- **`merger`** — merges fragments into a molecule and (optionally) predicts bonds. Defaults to `generator`; the core 15–39 model is recommended.

```python
import torch
from rdkit import Chem
from mlconfgen import MLConformerGenerator, inertial_fragment_matching, evaluate_samples

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fragment_generator = MLConformerGenerator(
    edm_weights="edm_moi_chembl_6_39_fragments.pt",
    adj_mat_seer_weights="adj_mat_seer_chembl_15_39.pt",
    diffusion_steps=100,
    device=device,
)

merger = MLConformerGenerator(
    edm_weights="edm_moi_chembl_15_39.pt",
    adj_mat_seer_weights="adj_mat_seer_chembl_15_39.pt",
    diffusion_steps=100,
    device=device,
)
```

---

## 1. IFM from a Reference Molecule

```python
reference = Chem.MolFromMolFile("./assets/demo_files/ceyyag.mol")

samples = inertial_fragment_matching(
    reference_conformer=reference,
    n_samples=20,
    generator=fragment_generator,
    merger=merger,
    variance=1,
    diffusion_steps_merging=10,
    predict_bonds=True,
    optimize_geometry=True,
)

_, results = evaluate_samples(reference, samples)
```

Workflow:

1. The reference is split by cutting single non-ring bonds until all fragments fall within `[min_frag_size, max_frag_size]` heavy atoms.
2. Each fragment is aligned to its own principal frame; shape-matched fragments are generated for all of them in one batch.
3. Generated fragments are rotated back and placed where the reference fragments were.
4. The assembled atom cloud is used as the seed of a short denoising run (`diffusion_steps_merging`) under the full reference context.
5. With `predict_bonds=True`, AdjMatSeer and standardisation are applied; disconnected results are rejected.

| Parameter | Default | Description |
|---|---|---|
| `reference_conformer` | — | Reference molecule with a 3D conformer. |
| `n_samples` | — | Number of molecules to generate. |
| `generator` / `merger` | — / `generator` | Fragment generator and merger models. |
| `variance` | `1` | ± heavy atoms around `n_atoms` for the final molecule. |
| `n_atoms` | reference size | Target median heavy-atom count. |
| `resample_steps` | `0` | Harmonising resampling steps for both stages. |
| `diffusion_steps_merging` | `10` | Denoising steps in the merge; ~10% of the merger's `diffusion_steps`. |
| `min_frag_size` / `max_frag_size` | `6` / `20` | Allowed fragment sizes when splitting. |
| `max_iter` | `200` | Attempts at finding a valid split before raising `RuntimeError`. |
| `predict_bonds` | `False` | Run AdjMatSeer + standardisation on the merged molecules. |
| `optimize_geometry` | `False` | MMFF94 optimisation during standardisation. |
| `verbose` | `False` | Log splitting progress. |

With `predict_bonds=False` the function returns atom clouds without bonds, suitable for `merger.predict_bonds(...)` or custom post-processing.

---

## 2. Fixed-Fragment IFM

`ff_inertial_fragment_matching` combines IFM with a fixed fragment. The MOI tensor of the fixed fragment is subtracted from the reference tensor to obtain the residual context of the part that still has to be generated. The generated fragment is positioned and merged with the fixed fragment, which is injected and preserved throughout the merge.

```python
from mlconfgen import ff_inertial_fragment_matching
from mlconfgen.utils import extract_fragment

ff_idx = {3, 5, 6, 7, 8, 9, 10}

# Fragment as indices of the reference ...
samples = ff_inertial_fragment_matching(
    fixed_fragment=ff_idx,
    reference_conformer=reference,
    generator=fragment_generator,
    merger=merger,
    n_samples=20,
    variance=1,
    predict_bonds=True,
    optimize_geometry=True,
)

# ... or as a molecule in the reference coordinate frame
fragment = extract_fragment(reference, ff_idx)
samples = ff_inertial_fragment_matching(
    fixed_fragment=fragment,
    reference_conformer=reference,
    generator=fragment_generator,
    merger=merger,
    n_samples=20,
)
```

### From an arbitrary context

Fixed-fragment IFM is the recommended way to fill an arbitrary shape (e.g. a pocket) around a known anchor fragment. The fragment must be a `Mol` expressed in the principal frame of the target shape:

```python
samples = ff_inertial_fragment_matching(
    fixed_fragment=fragment_in_principal_frame,
    reference_context=context,          # torch.Tensor of shape (3,)
    n_atoms=17,
    generator=fragment_generator,
    merger=merger,
    n_samples=20,
    variance=1,
    predict_bonds=True,
)
```

Additional parameters compared to plain IFM: `reference_context`, `blend_power` (fragment injection schedule, default `3`). The split-related parameters do not apply.

---

## 3. Recommendations

- IFM performs best with **reference molecules**; from arbitrary shapes it is less stable unless a fixed fragment anchors the generation.
- Keep `diffusion_steps_merging` at ~10% of the merger's total steps.
- Use `predict_bonds=True` and `evaluate_samples` to measure the shape gain over plain generation on your references.
- `inertial_fragment_matching` raises `RuntimeError` if the reference cannot be split within the size constraints (e.g. a single large ring system). Relax `max_frag_size` or fall back to basic generation.
