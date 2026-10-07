# Shape Similarity & Alignment

`mlconfgen.cheminformatics` implements a **Gaussian description of molecular shape** following Grant & Pickup [[1]](https://doi.org/10.1021/j100011a016)[[2]](https://doi.org/10.1007/978-94-017-1120-3_5). It provides two things the rest of the library relies on:

- a **shape quadrupole** (steric multipole) frame used to align molecules to each other without any atom correspondence;
- a **Shape Tanimoto** score based on the overlap of Gaussian molecular volumes.

Both are used by [`evaluate_samples`](1_molecule_generation.md#4-evaluating-samples) and by the fragment placement step of [Inertial Fragment Matching](5_inertial_fragment_matching.md).

---

## 1. Gaussian Molecular Volume

Every heavy atom \(i\) is represented by an isotropic 3D Gaussian

\[
g_i(\mathbf r) = p\,\exp\!\big(-\alpha\,|\mathbf r - \mathbf c_i|^2\big)
\]

with a common amplitude \(p = 2.70\) and a generic atom radius \(R = 1.60\) Å. The exponent is derived from the radius so that the Gaussian encloses the same volume as a hard sphere of radius \(R\) [[1]](https://doi.org/10.1021/j100011a016):

\[
\lambda = \frac{4\pi}{3p}, \qquad \kappa = \frac{\pi}{\lambda^{2/3}}, \qquad \alpha = \frac{\kappa}{R^2}
\]

Element-specific radii are deliberately **not** used — the score measures shape, not chemistry, and generated and reference molecules are compared on equal footing.

The volume of the molecule is the union of the atomic Gaussians, computed by inclusion–exclusion over products of overlapping Gaussians:

\[
V = \sum_i V_i - \sum_{i<j} V_{ij} + \sum_{i<j<k} V_{ijk} - \dots
\]

A product of \(n\) Gaussians with the same \(\alpha\) is again a Gaussian centred on the mean of the centres, with exponent \(n\alpha\) and a reduced amplitude; its integral and moments are therefore available in closed form. Terms are included up to **sixth order** (`n_terms=6`) and only for atoms that are mutually within the **neighbour threshold** (`2 × AMPLITUDE = 5.4` Å), which keeps the clique enumeration tractable for 15–39 heavy atoms.

---

## 2. Shape Quadrupole and the Shape Frame

`get_shape_quadrupole_for_molecule` computes the zeroth, first and second **steric moments** of the Gaussian volume [[1]](https://doi.org/10.1021/j100011a016):

1. **Volume** \(V\) (zeroth moment) and **centroid** (first moments divided by \(V\)); coordinates are re-centred on the centroid.
2. The **second-moment tensor** \(S_{ab} = \frac{1}{V}\int r_a r_b\,\rho(\mathbf r)\,d\mathbf r\) with the same inclusion–exclusion corrections.
3. \(S\) is diagonalised; the molecule is rotated into its eigenbasis so all off-diagonal moments vanish.
4. Axes are permuted so that the **XX moment is the largest**, i.e. the longest shape axis is \(x\), then \(y\), then \(z\).

```python
import torch
from rdkit import Chem
from mlconfgen.cheminformatics.shape_similarity import get_shape_quadrupole_for_molecule

mol = Chem.RemoveHs(Chem.MolFromMolFile("./assets/demo_files/ceyyag.mol"))
coord = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
coord = coord - coord.mean(dim=0)

main_moments, coord_in_shape_frame = get_shape_quadrupole_for_molecule(coord)
# main_moments: tensor([Sxx, Syy, Szz]) sorted descending
```

The returned `main_moments` are a compact, rotation-invariant shape descriptor; `coord_in_shape_frame` are the coordinates in the canonical **shape frame**.

### Resolving the sign ambiguity

Diagonalisation defines the axes only up to sign. Of the eight sign combinations, four are proper rotations: the identity and rotations by \(\pi\) about \(x\), \(y\) and \(z\). `best_pi_rotation_by_tanimoto` evaluates the Shape Tanimoto for each of the four and keeps the best:

```python
from mlconfgen.cheminformatics.shape_similarity import best_pi_rotation_by_tanimoto

best_coord, shape_tanimoto = best_pi_rotation_by_tanimoto(ref_coord_in_shape_frame, cand_coord_in_shape_frame)
```

Together, the shape-frame transform plus the best \(\pi\)-rotation constitute the **alignment** used throughout the library. It needs no atom mapping, is deterministic, and costs four grid evaluations per pair — a deliberate trade of the last few percent of overlap for speed and robustness compared to gradient-based optimisation of the overlap (as in ROCS [[3]](https://doi.org/10.1021/jm0603365)).

### Shape frame vs. inertial frame

The [MOI context](../model/1_architecture.md#shape-descriptor-moment-of-inertia) that conditions the EDM treats atoms as equal **point masses**, so its principal frame is cheap and exactly additive — the properties IFM exploits. The shape quadrupole treats atoms as **overlapping volumes**. The two frames are close for most molecules but not identical; the generator works in the inertial frame, while evaluation and sample alignment use the shape frame.

---

## 3. Shape Tanimoto Score

`tanimoto_score` compares two aligned point sets by the overlap of their Gaussian volumes [[2]](https://doi.org/10.1007/978-94-017-1120-3_5)[[4]](https://doi.org/10.1002/(SICI)1096-987X(19961115)17:14%3C1653::AID-JCC7%3E3.0.CO;2-K):

1. Both molecules are placed on a common **40 × 40 × 40 grid** whose bounds are the joint bounding box padded by \(6R\).
2. The molecular density at each grid point is the saturating union of atomic Gaussians,
   \(\rho(\mathbf r) = 1 - \prod_i \big(1 - g_i(\mathbf r)\big)\), which caps the density at 1 inside the molecule rather than letting overlapping atoms double-count.
3. Overlap integrals are evaluated as Riemann sums and combined into the Tanimoto coefficient

\[
T_{AB} = \frac{O_{AB}}{O_{AA} + O_{BB} - O_{AB}}, \qquad O_{XY} = \int \rho_X(\mathbf r)\,\rho_Y(\mathbf r)\,d\mathbf r
\]

\(T_{AB} \in (0, 1]\) and equals 1 for identical shapes.

```python
from mlconfgen.cheminformatics.shape_similarity import tanimoto_score

score = tanimoto_score(ref_coord, cand_coord)       # coordinates must already be aligned
score = tanimoto_score(ref_coord, cand_coord, n=60) # finer grid, slower
```

> **Note:** Hydrogens are excluded everywhere in the shape pipeline. All benchmark figures on the [Performance](../model/2_performance.md) page are heavy-atom Shape Tanimoto values obtained with this implementation at its default settings.

---

## 4. Where It Is Used

| Function | Role of shape similarity |
|---|---|
| `evaluate_samples(reference, samples)` | Reference and every sample are moved to their shape frames; each sample gets the best \(\pi\)-rotation and its `shape_tanimoto`. Returned mol blocks are aligned to the reference. |
| `inertial_fragment_matching(...)` | Generated fragments are brought to their inertial frame and the best \(\pi\)-rotation against the **reference fragment** is chosen by Shape Tanimoto before the fragment is placed back into the molecule. |
| `ff_inertial_fragment_matching(...)` | Same as above for the generated part when a reference fragment exists; with a `Mol` fixed fragment only the inertial-frame alignment is applied. |

Chemical similarity reported alongside (`chemical_tanimoto`) is the Tanimoto coefficient of 2-hop, 2048-bit Morgan fingerprints and is independent of the shape machinery.

---

## 5. Aligning Your Own Molecules

```python
import torch
from rdkit import Chem
from mlconfgen.utils import set_conformer_positions
from mlconfgen.cheminformatics.shape_similarity import (
    get_shape_quadrupole_for_molecule,
    best_pi_rotation_by_tanimoto,
)

def to_shape_frame(mol: Chem.Mol) -> torch.Tensor:
    mol = Chem.RemoveHs(mol)
    coord = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
    _, coord_sf = get_shape_quadrupole_for_molecule(coord - coord.mean(dim=0))
    return coord_sf

ref = Chem.MolFromMolFile("reference.mol")
cand = Chem.MolFromMolFile("candidate.mol")

ref_sf = to_shape_frame(ref)
cand_sf = to_shape_frame(cand)
best_cand_coord, score = best_pi_rotation_by_tanimoto(ref_sf, cand_sf)

aligned_ref = set_conformer_positions(Chem.RemoveHs(ref), ref_sf)
aligned_cand = set_conformer_positions(Chem.RemoveHs(cand), best_cand_coord)
print(f"Shape Tanimoto: {score:.3f}")
```

This is exactly what `evaluate_samples` does per sample; use it directly when you want aligned RDKit molecules rather than mol blocks, or want to compare molecules unrelated to a generation run.

---

## API Summary

`mlconfgen.cheminformatics.shape_similarity`

| Name | Description |
|---|---|
| `ATOM_RADIUS = 1.60`, `AMPLITUDE = 2.70`, `ALPHA` | Gaussian parameters shared by all functions. |
| `get_alpha(atom_radius, gaussian_amplitude)` | Exponent for a given radius/amplitude. |
| `get_shape_quadrupole_for_molecule(coordinates, amplitude, generic_atom_radius, n_terms=6, neighbour_threshold)` | `(main_moments, coordinates_in_shape_frame)`. Input must be centred. |
| `best_pi_rotation_by_tanimoto(ref_coord, cand_coord, tanimoto_fn=None)` | `(best_coord, best_score)` over identity and \(\pi\)-rotations about \(x,y,z\). |
| `tanimoto_score(ref_coord, cand_coord, alpha, amplitude, n=40)` | Shape Tanimoto of two aligned point sets on an \(n^3\) grid. |
| `Grid`, `torch_evaluate_density_on_grid(coordinates, grid, alpha, amplitude)` | Building blocks for custom overlap measures. |
| `product_of_n_gaussians`, `get_valid_combinations`, `find_r_cliques_fast` | Inclusion–exclusion helpers. |

`mlconfgen.cheminformatics`

| Name | Description |
|---|---|
| `evaluate_samples(reference, samples, generator, sanitize_ref=True)` | Alignment + shape and chemical Tanimoto for a batch of samples. |

---

## References

1. Grant, J. A.; Pickup, B. T. *A Gaussian Description of Molecular Shape.* J. Phys. Chem. **1995**, 99, 3503–3510. [10.1021/j100011a016](https://doi.org/10.1021/j100011a016)
2. Grant, J. A.; Pickup, B. T. *Gaussian Shape Methods.* In *Computer Simulation of Biomolecular Systems*, Vol. 3; Springer, **1997**. [10.1007/978-94-017-1120-3_5](https://doi.org/10.1007/978-94-017-1120-3_5)
3. Hawkins, P. C. D.; Skillman, A. G.; Nicholls, A. *Comparison of Shape-Matching and Docking as Virtual Screening Tools.* J. Med. Chem. **2007**, 50, 74–82. [10.1021/jm0603365](https://doi.org/10.1021/jm0603365)
4. Grant, J. A.; Gallardo, M. A.; Pickup, B. T. *A fast method of molecular shape comparison: A simple application of a Gaussian description of molecular shape.* J. Comput. Chem. **1996**, 17, 1653–1666. [10.1002/(SICI)1096-987X(19961115)17:14<1653::AID-JCC7>3.0.CO;2-K](https://doi.org/10.1002/(SICI)1096-987X(19961115)17:14%3C1653::AID-JCC7%3E3.0.CO;2-K)
5. Sapegin, D. et al. *Moment of inertia as a simple shape descriptor for diffusion-based shape-constrained molecular generation.* Digital Discovery **2025**. [10.1039/D5DD00318K](https://doi.org/10.1039/D5DD00318K)
