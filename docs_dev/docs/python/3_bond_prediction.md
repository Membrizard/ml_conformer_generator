# Bond Prediction

The EDM produces an **atom cloud** — element types and 3D coordinates, no bonds. Turning that into a molecule requires a bond-perception step, and MLConfGen supports two strategies:

| Strategy | Implementation | Validity (core model) | Character |
|---|---|---|---|
| **ML bond prediction** | AdjMatSeer GCN, built into `generate_conformers` | ~48% | Bond-order distribution learned from ChEMBL; chemically closer to the training set |
| **Deterministic bond prediction** | OpenBabel bond perception, user-side | ~93% | Rule-based from geometry; higher validity, more generic chemistry |

The ML path is what the library ships and what [Molecule Generation](1_molecule_generation.md) describes. This page shows how to plug the deterministic alternative into the pipeline.

> **Note:** OpenBabel is licensed under the **GPL-2.0**. `mlconfgen` is Apache 2.0 and therefore does **not** depend on or call OpenBabel anywhere in its code. The snippet below is an *example for your own environment*; by installing OpenBabel you accept its license terms for your project.

---

## 1. Why Two Strategies

AdjMatSeer is trained to reproduce ChEMBL bonding patterns from distances, so the molecules it produces have a chemical-feature distribution close to drug-like compounds — at the cost of a lower validity rate, since a single wrong bond order fails sanitisation. OpenBabel's perception (`ConnectTheDots` + `PerceiveBondOrders`) is purely geometric and fails far less often, but it does not know what *typical* chemistry looks like.

The **distilled AdjMatSeer** (`small_adj_mat_seer_obabel_15_39.pt` / `.onnx`) was trained to imitate OpenBabel's labels and reaches ~71% validity. It is the way to get OpenBabel-grade bonds in environments where OpenBabel cannot run — the ONNX backend and the [JavaScript package](../javascript/1_quick_start.md).

---

## 2. Installation

```bash
pip install openbabel-wheel      # pre-built wheels
# or
conda install -c conda-forge openbabel
```

---

## 3. Example: Deterministic Bond Prediction

The EDM stage is called directly (`edm_samples`), OpenBabel assigns the bonds, and the library's own `standardize_mol` finishes the job exactly as it would after AdjMatSeer.

```python
from openbabel import openbabel
from rdkit import Chem
from mlconfgen import MLConformerGenerator
from mlconfgen.utils import standardize_mol

ob.obErrorLog.SetOutputLevel(0)   # silence OpenBabel warnings

BOND_TYPES = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}


def guess_bonds_openbabel(mol: Chem.Mol) -> Chem.Mol:
    ob_conv = openbabel.OBConversion()
    ob_conv.SetInAndOutFormats("xyz", "mol")
    obmol = openbabel.OBMol()
    xyz_block = Chem.MolToXYZBlock(mol)
    ob_conv.ReadString(obmol, xyz_block)

    obmol.ConnectTheDots()
    obmol.PerceiveBondOrders()

    mol_block = ob_conv.WriteString(obmol)
    raw_mol = Chem.MolFromMolBlock(mol_block)
    if raw_mol:
        out_mol = strip_mol(raw_mol)
    else:
        out_mol = None



model = MLConformerGenerator(diffusion_steps=100)
reference = Chem.MolFromMolFile("./assets/demo_files/ceyyag.mol")

# Stage 1: atom clouds from the EDM
ref_context, ref_n_atoms, _ = model.prepare_inputs(reference_conformer=reference)

edm_mols = model.edm_samples(
    reference_context=ref_context,
    n_samples=50,
    min_n_nodes=ref_n_atoms - 2,
    max_n_nodes=ref_n_atoms + 2,
)

# Stage 2: deterministic bonds instead of AdjMatSeer
bonded = [guess_bonds_openbabel(m) for m in edm_mols]

# Stage 3: the library's standardisation (largest fragment, valence, kekulisation, MMFF94)
samples = [
    std for m in bonded
    if m is not None and (std := standardize_mol(m, optimize_geometry=True)) is not None
]

print(f"{len(samples)} / {len(edm_mols)} valid")
```

Everything downstream — `evaluate_samples`, RL scoring functions, export — accepts these molecules like any other output of `generate_conformers`.

### Notes on the example

- **Fixed fragments.** Atom clouds from inpainting (`edm_samples(..., fixed_fragment=...)`) are processed the same way — only the geometry is used.
- **ONNX backend.** `MLConformerGeneratorONNX.edm_samples` returns the same kind of bond-less molecules, so the snippet works unchanged with `MLConformerGeneratorONNX` in place of `MLConformerGenerator`.

---

## 4. Choosing a Strategy

- Prefer **AdjMatSeer** when chemical plausibility relative to ChEMBL matters (lead-like libraries, downstream chemical-similarity scoring) and you can afford to over-sample.
- Prefer **OpenBabel** when throughput of valid molecules is the goal, when geometry is the only thing you trust, or when comparing against the 93% figure on the [Performance](../model/2_performance.md) page.
- Prefer the **distilled AdjMatSeer** when you want OpenBabel-like validity without a GPL dependency or in ONNX / browser deployments.
- Fine-tuning checkpoints update the AdjMatSeer head; with a checkpoint loaded, use the ML path so the fine-tuned bond predictor is applied.
