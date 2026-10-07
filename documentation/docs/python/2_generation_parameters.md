# Generation Parameters
*(Detailed Guide)*

This document explains the parameters of `MLConformerGenerator` and `generate_conformers`, their meaning and their effect on generation. `MLConformerGeneratorONNX` accepts the same generation parameters with `numpy.ndarray` in place of `torch.Tensor`.

---

## Constructor

```python
MLConformerGenerator(
    diffusion_steps=100,
    device="cpu",
    edm_weights="edm_moi_chembl_15_39.pt",
    adj_mat_seer_weights="adj_mat_seer_chembl_15_39.pt",
    finetune_checkpoint=None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `diffusion_steps` | int | `100` | Number of denoising steps (max 1000). |
| `device` | `torch.device` / str | `"cpu"` | Device the models run on. |
| `edm_weights` | str / Path | core EDM | Path or Hugging Face file name of the EDM state dict. |
| `adj_mat_seer_weights` | str / Path | core AdjMatSeer | Path or Hugging Face file name of the AdjMatSeer state dict. |
| `finetune_checkpoint` | str / Path | `None` | RL fine-tuning checkpoint to load at start-up. |
| `min_n_nodes` / `max_n_nodes` | int | `6` / `39` | Hard clamp on heavy-atom counts the generator will request. |
| `dimension` | int | `42` | Padded tensor size; must match the weights. |
| `context_norms`, `atom_decoder`, `num_bond_types` | | | Model constants; override only for custom-trained weights. |

Model width and base timesteps are read from the checkpoints, so distilled and full weights load through the same constructor.

---

## `generate_conformers`

```python
samples = model.generate_conformers(
    reference_conformer=None,
    n_samples=10,
    variance=2,
    reference_context=None,
    n_atoms=None,
    optimize_geometry=True,
    resample_steps=0,
    fixed_fragment=None,
    blend_power=3,
    keep_largest_fragment=True,
)
```

| Parameter | Type | Default | Required | Description |
|---|---|---|---|---|
| `reference_conformer` | `Chem.Mol` | `None` | ✅ one of | Reference molecule with a 3D conformer. |
| `reference_context` | `torch.Tensor (3,)` | `None` | ✅ one of | Principal MOI components instead of a molecule. |
| `n_atoms` | int | `None` | ⚠️ with context | Target heavy-atom count when generating from a context. |
| `n_samples` | int | `10` | | Number of samples to draw. |
| `variance` | int | `2` | | Allowed ± deviation in heavy-atom count. |
| `optimize_geometry` | bool | `True` | | Constrained MMFF94 optimisation during standardisation. |
| `resample_steps` | int | `0` | | Extra resampling per denoising step. |
| `fixed_fragment` | `Chem.Mol` / `set[int]` | `None` | | Substructure to keep fixed (inpainting). |
| `blend_power` | int | `3` | | Polynomial blending schedule for the fixed fragment. |
| `keep_largest_fragment` | bool | `True` | | Keep the largest connected component, or discard disconnected samples. |

---

## Detailed Parameter Explanations

### `reference_conformer`

An RDKit `Mol` with 3D coordinates. Hydrogens are removed internally; the heavy-atom count becomes the reference size. The molecule is aligned to its principal inertial frame and its MOI eigenvalues are used as the generation context.

- Flat conformers (all-zero z-coordinates) yield degenerate contexts — generate a 3D conformer first.
- Shapes with more than ~40 heavy atoms exceed the training range; consider [Inertial Fragment Matching](5_inertial_fragment_matching.md) or a context with an explicit `n_atoms`.

---

### `reference_context` and `n_atoms`

The three principal moments of inertia of the target shape (equal-mass points, ascending order), and the heavy-atom count to aim for. `n_atoms` is mandatory with a context because the shape alone carries no size information.

```python
from mlconfgen.utils import align_mol_to_principal_frame

context, shift, rotation, aligned_coord = align_mol_to_principal_frame(reference)
```

A `set` of atom indices cannot be used as `fixed_fragment` together with a context (there is no reference molecule to index into); pass a `Mol` instead.

---

### `n_samples`

Number of candidates drawn from the EDM. Only chemically valid, standardised molecules are returned:

- **Core AdjMatSeer**: ~50% of samples survive.
- **Distilled AdjMatSeer (`small_adj_mat_seer_obabel_15_39.pt`)**: ~70% survive.
- **Deterministic (OpenBabel) bond prediction**: ~93% survive; see [Bond Prediction](3_bond_prediction.md).

---

### `variance`

Each sample's heavy-atom count is drawn uniformly from `[n_ref − variance, n_ref + variance]`, clamped to `[min_n_nodes, max_n_nodes]`. Use `variance=0` for a fixed size; `1–2` adds structural diversity while staying close to the reference.

---

### `optimize_geometry`

When `True`, standardisation ends with a constrained MMFF94 optimisation that relaxes bond lengths and angles while keeping the generated pose. Set to `False` if geometry is re-optimised later in your pipeline — it is the most expensive post-processing step.

---

### `resample_steps`

Number of additional forward/backward resampling passes per denoising step (RePaint-style harmonisation). Improves validity and shape similarity by roughly 5–10% at a proportional cost in time. Most useful with a `fixed_fragment`; recommended `4–10` there, `0` otherwise.

---

### `fixed_fragment` and `blend_power`

Enables inpainting: atoms of the fragment are kept in place while the rest of the molecule is generated around them. See [Fixed Fragments](4_fixed_fragments.md) for details and coordinate-frame rules.

`blend_power` controls the schedule `(1 − t)^blend_power` with which the known fragment is injected into the denoising trajectory. `0` is a hard injection; higher values give smoother blending (default `3`).

---

### `keep_largest_fragment`

The EDM occasionally produces disconnected atom groups. By default the largest connected component is kept and the rest discarded. With `False`, any disconnected sample is rejected entirely — this is the behaviour used in the IFM merging step.

---

### `diffusion_steps` (constructor)

- `100` is typically sufficient; `1000` gives little additional benefit.
- `20–50` speeds generation up roughly linearly with moderate quality loss.
- Fine-tuning is commonly run at `10` steps for speed; see [RL Fine-Tuning](7_fine_tuning.md).

---

## Practical Tips

- Start simple: `reference_conformer` + `n_samples`.
- Keep `optimize_geometry=True` unless you post-process geometry yourself.
- For fixed-fragment workflows, raise `resample_steps` to `4–10` and keep `blend_power` at `3`.
- For reproducibility seed `torch` (and `numpy` for the ONNX backend) before generation.
- The generator is a `torch.nn.Module`; `model(reference_conformer=..., n_samples=...)` is equivalent to `generate_conformers`.
