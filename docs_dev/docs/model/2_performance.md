# Performance

All figures are measured at **100 denoising steps** on **100,000 samples** generated from the 1,000 CCDC Virtual Screening [[4]](https://www.ccdc.cam.ac.uk/support-and-resources/downloads/) reference compounds, in batches of 100 samples.

---

## Core Model

*24M EDM + 22M AdjMatSeer parameters*

| Metric | Value |
|---|---|
| Avg time to generate 50 valid samples | 11.46 s (NVIDIA H100) |
| Generation speed | 4.18 valid molecules / s |
| GPU memory per generation thread | up to 14.0 GB (`float16`, 39 atoms, 100 samples) |
| Avg Shape Tanimoto similarity | 53.32% (basic) — 69.97% (Inertial Fragment Matching) |
| Max Shape Tanimoto similarity | 99.69% |
| Avg chemical Tanimoto similarity (Morgan r=2, 2048 bits) | 10.87% |
| Avg fragment integration success (IFM) | 63.80% |
| Chemically novel vs. training set | 99.84% |
| Valid after standardisation | 48% (ML bond prediction) — 93% (OpenBabel bond prediction) |
| Unique in generated set | 99.94% |
| Fréchet fingerprint distance | ChEMBL 4.13 · PubChem 2.64 · ZINC-250k 4.95 |

---

## Distilled Model

*8.9M EDM + 1.6M AdjMatSeer parameters*

> **Note:** AdjMatSeer distillation was accompanied by fine-tuning to match OpenBabel bond perception to maximise validity. This may affect the general chemical quality of the generated structures.

| Metric | Value |
|---|---|
| Avg time to generate 50 valid samples | 10.9 s (NVIDIA A40) |
| Generation speed | 4.59 valid molecules / s |
| GPU memory per generation thread | up to 8.6 GB (`float16`, 39 atoms, 100 samples) |
| Avg Shape Tanimoto similarity | 50.5% |
| Max Shape Tanimoto similarity | 88.65% |
| Avg chemical Tanimoto similarity (Morgan r=2, 2048 bits) | 7.53% |
| Valid after standardisation | 71.14% (ML bond prediction) |

Choose the distilled weights when memory or throughput is the constraint, or when validity under ML bond prediction matters more than chemical fidelity to ChEMBL. Choose the core weights for maximum shape similarity and IFM workflows.

---

## Generation Quality

### PoseBusters validity [[5]](https://doi.org/10.1039/D3SC04185A)

**PB-valid molecules: 91.33%**

| Check | Failure rate |
|---|---|
| position | 0.01% |
| mol_pred_loaded | 0.0% |
| sanitization | 0.01% |
| inchi_convertible | 0.01% |
| all_atoms_connected | 0.0% |
| bond_lengths | 0.24% |
| bond_angles | 0.70% |
| internal_steric_clash | 2.31% |
| aromatic_ring_flatness | 3.34% |
| non-aromatic_ring_non-flatness | 0.27% |

### Synthesizability

**Average SA Score [[6]](https://doi.org/10.1186/1758-2946-1-8): 3.18** (scale 1 = easy to 10 = very difficult)

<img src="https://raw.githubusercontent.com/Membrizard/ml_conformer_generator/main/assets/benchmarks/sa_score_dist.png" width="300">

---

## Practical Scaling Notes

- Runtime scales linearly with `diffusion_steps` and roughly linearly with `n_samples` until the device saturates.
- Memory is dominated by the EDM's edge tensors and scales with `n_samples × max_n_nodes²`; reduce the batch for large molecules on small GPUs.
- `resample_steps = k` multiplies the denoising cost by approximately `k + 1`.
- MMFF94 optimisation (`optimize_geometry=True`) runs on CPU per molecule and becomes noticeable for batches of hundreds.

---

## Generation Examples

![ex1](https://raw.githubusercontent.com/Membrizard/ml_conformer_generator/main/assets/ref_mol/molecule_1.png)
![ex2](https://raw.githubusercontent.com/Membrizard/ml_conformer_generator/main/assets/ref_mol/molecule_2.png)
![ex3](https://raw.githubusercontent.com/Membrizard/ml_conformer_generator/main/assets/ref_mol/molecule_3.png)
![ex4](https://raw.githubusercontent.com/Membrizard/ml_conformer_generator/main/assets/ref_mol/molecule_4.png)

---

## References

4. CCDC Virtual Screening set — [ccdc.cam.ac.uk](https://www.ccdc.cam.ac.uk/support-and-resources/downloads/)
5. Buttenschoen et al., *PoseBusters* — [10.1039/D3SC04185A](https://doi.org/10.1039/D3SC04185A)
6. Ertl & Schuffenhauer, *SA Score* — [10.1186/1758-2946-1-8](https://doi.org/10.1186/1758-2946-1-8)
