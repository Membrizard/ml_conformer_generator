# ML Conformer Generator

**ML Conformer Generator** is a shape-constrained molecule generation model that combines
an Equivariant Diffusion Model (EDM) and Graph Convolutional Network (GCN). It generates 3D conformations
that are chemically valid and geometrically aligned with a reference shape.

---

## 📦 Model Summary

- **Developed by:** Denis Sapegin
- **Architecture**: Equivariant Diffusion Model (EDM) + Graph Convolutional Network (GCN)
- **Training Data**: 1.6 million ChEMBL compounds, filtered for molecules with 15–39 heavy atoms
- **Elements Supported**: H, C, N, O, F, P, S, Cl, Br
- **Post-Processing**: Deterministic standardization pipeline using RDKit with constrained MMFF94 geometry optimization
- **Primary Metric**: Shape Tanimoto Similarity

---

## 🚀 Intended Use

- Non-Commercial Research in 3D molecular generation
- Academic/educational use
- 

---

## 🚫 Out of Scope / Limitations

- Not licensed for **commercial use** without explicit permission
- Trained on ChEMBL, so results may be biased towards drug-like chemistry
- 

---

## 💾 Access & Licensing

The **Python package and inference code are available on GitHub** 

> The model weights provided are under non-commercial license.

For commercial licensing and inference-as-a-service, contact:
dasapegin@gmail.com

---

## 🧪 Evaluation Metrics (100,000 requested samples, 100 denoising steps)

- ✅ **Valid molecules (post-standardization, % from requested)**: 48%
- 🧬 **Chemical novelty**: 99.84%
- 📐 **Avg Shape Tanimoto**: 53.32%
- 🎯 **Max Shape Tanimoto**: 99.69%
- 🔁 **Unique molecules**: 99.94%
- ⚡ **Generation speed**: 4.18 valid molecules/sec (NVIDIA H100)
- 💾 **Memory (per thread)**: up to 4.0 GB
- 🧬 **Fréchet Fingerprint Distance (to ChEMBL)**: 4.13

---

## 🧠 How It Works

### Core Components:
- **EDM** generates atom coordinates and types under shape constraints
- **GCN** predicts adjacency matrices (bonding)
- **RDKit** pipeline enforces valence, performs sanitization, and optimizes geometry

### Shape Alignment:
Evaluated using **Gaussian molecular volume overlap** and **Shape Tanimoto Similarity**.

Hydrogens are excluded from similarity computation.

---

## 🐍 API Usage (Python)

```python
from rdkit import Chem
from ml_conformer_generator import MLConformerGenerator

model = MLConformerGenerator(diffusion_steps=100)
reference = Chem.MolFromMolFile('MOL_FILE_NAME.mol')
samples = model.generate_conformers(reference_conformer=reference, n_samples=20)
```
Supports ONNX export and inference for PyTorch-free runtime.

## Export to ONNX

Convert the model to ONNX for runtime flexibility:
```python
from ml_conformer_generator import MLConformerGenerator

generator = MLConformerGenerator()
generator.export_to_onnx()
```
This will compiles and saves the models to:
`./ml_conformer_generator/ml_conformer_generator/weights/`

## ONNX Inference:

After the export is complete the PyTorch-free interface for the model can be used:

```python
from ml_conformer_generator import MLConformerGeneratorONNX
from rdkit import Chem

generator = MLConformerGeneratorONNX(diffusion_steps=100)
reference = Chem.MolFromMolFile('MOL_FILE_NAME.mol')
samples = generator.generate_conformers(reference_conformer=reference, n_samples=20)
```