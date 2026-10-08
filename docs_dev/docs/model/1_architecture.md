# Architecture & Pipeline

---

## Components

The generator combines three parts:

- **Equivariant Diffusion Model (EDM)** [[1]](https://doi.org/10.48550/arXiv.2203.17003) — an E(3)-equivariant graph neural network (EGNN) trained as a denoising diffusion model. It generates heavy-atom coordinates and element types jointly, conditioned on the reference shape.
- **AdjMatSeer** — a Graph Convolutional Network [[2]](https://doi.org/10.1039/D3DD00178D) that predicts the full bond-order adjacency matrix (none / single / double / triple / aromatic) from element types and the inter-atomic distance matrix.
- **Deterministic standardisation pipeline** — RDKit-based clean-up and validation of the predicted molecule.

---

## Shape Descriptor: Moment of Inertia

The reference shape is encoded by the three principal components of its **Moment of Inertia (MOI) tensor**, computed for equal-mass heavy atoms (or arbitrary points) after centering. The reference is rotated into its principal inertial frame; the resulting eigenvalues are normalised with dataset statistics (`CONTEXT_NORMS`) and broadcast to every node as the EDM conditioning context.

Properties that the pipeline relies on:

- **Rotation/translation invariance** — the descriptor does not depend on the pose of the reference.
- **Additivity** — the MOI tensor of a molecule equals the sum of the tensors of its parts (about the common centre). This is the basis of [Inertial Fragment Matching](../python/5_inertial_fragment_matching.md): subtracting a fixed fragment's tensor from the reference yields the context of the part still to be generated.
- **Size agnostic** — the descriptor alone does not fix the number of atoms; `n_atoms` is a separate input.

Generated molecules come out in the principal frame of the reference. `evaluate_samples` (Python) and the Python inference pipeline undo the transform when aligning samples back to the original reference coordinates.

---

## Generation Pipeline

1. **Input preparation** — strip hydrogens, align the reference to its principal frame, compute the context; sample a heavy-atom count per molecule in `[n_ref − variance, n_ref + variance]`; build node and edge masks.
2. **Denoising** — start from Gaussian noise with zero centre of mass and run `diffusion_steps` reverse steps of the EDM (optionally with resampling and fixed-fragment injection). The output is a coordinate tensor `x` and a one-hot element tensor `h`.
3. **Canonical atom ordering** — a first-order connectivity is guessed from distances, and atoms are reordered by RDKit's canonical SMILES output order. AdjMatSeer was trained on canonically ordered inputs, and the JavaScript package reproduces the same ordering through RDKit.js.
4. **Bond prediction** — AdjMatSeer receives padded `elements` (B × 42), `dist_mat` (B × 42 × 42) and the binary connectivity guess, and returns bond-type logits (B × 42 × 42 × 5). The argmax over bond types defines the bonds. A deterministic alternative is described in [Bond Prediction](../python/3_bond_prediction.md).
5. **Standardisation**:
    - largest connected fragment (or rejection of disconnected molecules)
    - valence check
    - kekulisation
    - RDKit sanitisation
    - constrained MMFF94 geometry optimisation (Python only)

Molecules failing any step are dropped, which is why the number of returned molecules is below `n_samples`.

---

## Training Data

- **1.6 million** compounds from **ChEMBL**
- Filtered to **15–39 heavy atoms**
- Supported elements: `H, C, N, O, F, P, S, Cl, Br` (hydrogens are implicit at generation time)
- The fragment model `edm_moi_chembl_6_39_fragments.pt` extends the range down to 6 heavy atoms for IFM

---

## Model Variants

| Variant | EDM | AdjMatSeer | Notes |
|---|---|---|---|
| Core | 23.9M params, hidden width 420 | 21.8M params, hidden width 2048 | Reference quality |
| Distilled | 8.9M params, hidden width 256 | 1.6M params, hidden width 512 | Width-distilled from the core; AdjMatSeer additionally fine-tuned to match OpenBabel bond perception for higher validity |

The Python generator reads the hidden width and the base timestep count from the checkpoint, so both variants load through the same constructor. The same applies to the exported ONNX graphs.

---

## Evaluation Pipeline

`evaluate_samples` aligns each sample to the reference and reports:

- **Shape Tanimoto similarity** [[3]](https://doi.org/10.1007/978-94-017-1120-3_5) via Gaussian molecular volume overlap. Reference and sample are each brought into the principal frame of their **shape quadrupole**, and the score is maximised over the four π-rotations of that frame. Hydrogens are ignored.
- **Chemical Tanimoto similarity** of 2-hop, 2048-bit Morgan fingerprints.

The Gaussian shape model and the alignment procedure are described in detail in [Shape Similarity & Alignment](../python/6_shape_similarity.md).

---

## RL Fine-Tuning Architecture

Fine-tuning keeps the base EDM frozen and trains:

- an **EDM adapter** operating on the EDM outputs `(x, h)` under the node/edge masks;
- the **AdjMatSeer output head** (`resize`), with bond matrices sampled from the logits at a configurable temperature.

The policy-gradient loss combines the clipped reward, an adapter term weighted by `lambda_edm_adapter`, and a regulariser weighted by `lambda_edm_reg`. Checkpoints contain only these two components and remain small. See [RL Fine-Tuning](../python/7_fine_tuning.md).

---

## References

1. Hoogeboom et al., *Equivariant Diffusion for Molecule Generation in 3D* — [arXiv:2203.17003](https://doi.org/10.48550/arXiv.2203.17003)
2. *Graph convolutional bond prediction* — [10.1039/D3DD00178D](https://doi.org/10.1039/D3DD00178D)
3. Shape Tanimoto similarity — [10.1007/978-94-017-1120-3_5](https://doi.org/10.1007/978-94-017-1120-3_5)
4. Sapegin et al., *Moment of inertia as a simple shape descriptor for diffusion-based shape-constrained molecular generation*, Digital Discovery 2025 — [10.1039/D5DD00318K](https://doi.org/10.1039/D5DD00318K)
