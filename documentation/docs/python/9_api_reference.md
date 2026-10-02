# API Reference

Public surface of the `mlconfgen` package. Signatures are abbreviated to their meaningful arguments; see the per-topic pages for parameter semantics.

```python
from mlconfgen import (
    MLConformerGenerator,
    MLConformerGeneratorONNX,
    evaluate_samples,
    inertial_fragment_matching,
    ff_inertial_fragment_matching,
)
```

---

## `MLConformerGenerator`

`torch.nn.Module`. PyTorch generation pipeline.

```python
MLConformerGenerator(
    diffusion_steps: int = 100,
    device: torch.device | str = "cpu",
    edm_weights: str | Path = "edm_moi_chembl_15_39.pt",
    adj_mat_seer_weights: str | Path = "adj_mat_seer_chembl_15_39.pt",
    finetune_checkpoint: str | Path | None = None,
    dimension: int = 42,
    num_bond_types: int = 5,
    min_n_nodes: int = 6,
    max_n_nodes: int = 39,
    context_norms: dict = CONTEXT_NORMS,
    atom_decoder: dict = ATOM_DECODER,
)
```

Attributes: `generative_model` (`EquivariantDiffusion`), `adj_mat_seer` (`AdjMatSeer`), `edm_adapter` (`EDMAdapter | None`), `weights_manager`, `context_norms`, `device`.

### Methods

#### `generate_conformers(...) -> list[Chem.Mol]`

```python
generate_conformers(
    reference_conformer: Chem.Mol = None,
    n_samples: int = 10,
    variance: int = 2,
    reference_context: torch.Tensor = None,
    n_atoms: int = None,
    optimize_geometry: bool = True,
    resample_steps: int = 0,
    fixed_fragment: Chem.Mol | set = None,
    blend_power: int = 3,
    keep_largest_fragment: bool = True,
)
```

Full pipeline: EDM → AdjMatSeer → standardisation. Returns valid molecules only. `forward(...)` / `model(...)` is an alias without `keep_largest_fragment`.

#### `edm_samples(...) -> list[Chem.Mol] | tuple[Tensor, ...]`

```python
edm_samples(
    reference_context: torch.Tensor,
    n_samples: int = 100,
    max_n_nodes: int = 32,
    min_n_nodes: int = 25,
    resample_steps: int = 0,
    fixed_fragment: Chem.Mol = None,
    blend_power: int = 3,
    raw_output: bool = False,
)
```

Diffusion stage only. Returns bond-less molecules, or `(x, h, node_mask, edge_mask)` when `raw_output=True`. Applies the EDM adapter if a checkpoint is loaded. `reference_context` must already be the un-normalised MOI context (see `prepare_inputs`).

#### `predict_bonds(edm_samples: list[Chem.Mol]) -> list[Chem.Mol]`

Canonicalises atom order, runs AdjMatSeer and writes the predicted bonds into the molecules. No standardisation.

#### `prepare_inputs(...) -> tuple[Tensor, int, Chem.Mol | None]` *(static)*

```python
prepare_inputs(reference_conformer=None, fixed_fragment=None, reference_context=None, n_atoms=None)
```

Returns `(reference_context, reference_n_atoms, prepared_fixed_fragment)`. Raises `ValueError` if neither a conformer nor a context + `n_atoms` is given, or if a `set` fragment is combined with a context.

#### `fine_tune(...) -> None`

See [RL Fine-Tuning](7_fine_tuning.md) for the full parameter list. Writes `best_checkpoint.pt` / `latest_checkpoint.pt` into `save_dir`.

#### `load_finetune_checkpoint(path: str | Path) -> None`

Loads the EDM adapter and AdjMatSeer head from an RL checkpoint.

#### `list_weights() -> dict[str, list[str]]`

`{"remote": [...], "local": [...]}` of `.pt` files available on Hugging Face and in the cache.

#### `clear_cache() -> None`

Deletes the weights cache directory.

#### `random(size: int = 1, seed=None, optimize_geometry: bool = True) -> list[Chem.Mol]`

Generates molecules from a random context drawn from a bundled set of ChEMBL-derived seeds (`random_molecule_seeds.json`, 15–39 heavy atoms). Useful for smoke tests and unconditional sampling.

---

## `MLConformerGeneratorONNX`

Torch-free generation pipeline on ONNX Runtime.

```python
MLConformerGeneratorONNX(
    diffusion_steps: int = 100,
    egnn_onnx: str | Path = "egnn_chembl_15_39.onnx",
    adj_mat_seer_onnx: str | Path = "adj_mat_seer_chembl_15_39.onnx",
    finetune_checkpoint_onnx: str | Path | None = None,
    dimension: int = 42,
    min_n_nodes: int = 6,
    max_n_nodes: int = 39,
    context_norms: dict = CONTEXT_NORMS,
    atom_decoder: dict = ATOM_DECODER,
)
```

Methods: `generate_conformers`, `edm_samples` (no `raw_output`), `predict_bonds`, `prepare_inputs`, `list_weights` (`.onnx` files), `clear_cache`, `random`, `__call__`. Contexts are `numpy.ndarray` of shape `(3,)`. Attributes `generative_model` (`EquivariantDiffusionONNX`), `adj_mat_seer` and `edm_adapter` are `onnxruntime.InferenceSession` objects.

---

## `evaluate_samples`

```python
evaluate_samples(
    reference: Chem.Mol,
    samples: list[Chem.Mol],
    generator: rdFingerprintGenerator = <Morgan r=2, 2048 bits>,
    sanitize_ref: bool = True,
) -> tuple[str, list[dict]]
```

Returns the reference mol block in its shape-quadrupole frame and, per sample, `{"mol_block", "shape_tanimoto", "chemical_tanimoto"}` with the sample aligned to the reference. Hydrogens are ignored.

### `mlconfgen.cheminformatics.shape_similarity`

| Name | Description |
|---|---|
| `get_shape_quadrupole_for_molecule(coordinates, ...)` | `(main_moments, coordinates_in_shape_frame)` of a centred point set. |
| `best_pi_rotation_by_tanimoto(ref_coord, cand_coord)` | `(best_coord, best_score)` over identity and π-rotations about x, y, z. |
| `tanimoto_score(ref_coord, cand_coord, n=40)` | Gaussian-volume Shape Tanimoto of two aligned point sets. |
| `ATOM_RADIUS`, `AMPLITUDE`, `ALPHA`, `get_alpha(...)` | Gaussian shape parameters. |

See [Shape Similarity & Alignment](6_shape_similarity.md).

---

## `inertial_fragment_matching`

```python
inertial_fragment_matching(
    reference_conformer: Chem.Mol,
    n_samples: int,
    generator: MLConformerGenerator,
    merger: MLConformerGenerator = None,
    variance: int = 1,
    n_atoms: int = None,
    resample_steps: int = 0,
    diffusion_steps_merging: int = 10,
    min_frag_size: int = 6,
    max_frag_size: int = 20,
    max_iter: int = 200,
    verbose: bool = False,
    predict_bonds: bool = False,
    optimize_geometry: bool = False,
) -> list[Chem.Mol]
```

## `ff_inertial_fragment_matching`

```python
ff_inertial_fragment_matching(
    fixed_fragment: Chem.Mol | set,
    n_samples: int,
    generator: MLConformerGenerator,
    reference_conformer: Chem.Mol = None,
    reference_context: torch.Tensor = None,
    n_atoms: int = None,
    merger: MLConformerGenerator = None,
    variance: int = 1,
    resample_steps: int = 0,
    blend_power: int = 3,
    diffusion_steps_merging: int = 10,
    predict_bonds: bool = False,
    optimize_geometry: bool = False,
) -> list[Chem.Mol]
```

See [Inertial Fragment Matching](5_inertial_fragment_matching.md).

---

## `mlconfgen.rl_fine_tuning`

- `RLFineTuner` — the trainer used by `fine_tune`; usable directly for custom sampling functions.
- `EDMAdapter` — the trainable adapter applied to EDM outputs.
- `reinvent_score_wrapper.ReinventScoreWrapper(config_path, fmt=Format.TOML)` — REINVENT4 scoring as a scoring function.

---

## `mlconfgen.utils`

Frequently useful helpers (all importable from `mlconfgen.utils`):

| Name | Description |
|---|---|
| `WeightsManager(cache_dir=None)` | `resolve(filename, force_download=False)`, `clear_cache()`, `list_available_weights(suffixes)`. |
| `align_mol_to_principal_frame(mol)` | `(context, shift, rotation, aligned_coord)` for a molecule. |
| `get_moment_of_inertia_tensor(coord)` | MOI tensor of a point cloud. |
| `apply_transform(coord, shift, rotation)` / `inverse_coord_transform(...)` | Apply / undo the principal-frame transform. |
| `extract_fragment(mol, atom_idx: set)` | Sub-molecule with coordinates. |
| `split_molecule_size_constrained(mol, min_size, max_size, ...)` | Fragment sets used by IFM. |
| `set_conformer_positions(mol, coord)` | Write coordinates into a molecule's conformer. |
| `canonicalise(mol)` | Reorder atoms canonically (the order AdjMatSeer expects). |
| `standardize_mol(mol, optimize_geometry=True, ifm_mode=False)` | Standardisation pipeline; returns `None` on failure. |
| `is_valid_mol(mol)` | `1.0` / `0.0` sanitisation check. |
| `samples_to_rdkit_mol(positions, one_hot, node_mask, atom_decoder)` | Raw EDM tensors → bond-less molecules. |
| `random_context(size=1, seed=None)` | Random `{"context": [...], "n_atoms": n}` seed. |

Constants: `DIMENSION = 42`, `NUM_BOND_TYPES = 5`, `MIN_N_NODES = 6`, `MAX_N_NODES = 39`, `MIN_FRAG_SIZE = 6`, `MAX_FRAG_SIZE = 20`, `CONTEXT_NORMS`, `ATOM_DECODER` (`C, N, O, F, P, S, Cl, Br`).

---

## `onnx_export` (repository package)

```python
export_to_onnx(
    model: MLConformerGenerator,
    egnn_save_path="./egnn_chembl_15_39.onnx",
    adj_mat_seer_save_path="./adj_mat_seer_chembl_15_39.onnx",
    edm_adapter_save_path="./finetune_checkpoint.onnx",
    report: bool = False,
) -> None
```

See [ONNX Inference & Export](8_onnx.md).
