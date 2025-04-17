# ML Conformer Generator

<img src="https://raw.githubusercontent.com/Membrizard/ml_conformer_generator/main/assets/logo/mlconfgen_logo.png" width="200" style="display: block; margin: 0 10%;">

**ML Conformer Generator** 
is a tool for shape-constrained molecule generation using an Equivariant Diffusion Model (EDM)
and a Graph Convolutional Network (GCN). It is designed to generate 3D molecular conformations
that are both chemically valid and spatially similar to a reference shape.

---
## Installation

1. Install the package:

`pip install mlconfgen`

2. Load the weights from Huggingface
> https://huggingface.co/Membrizard/ml_conformer_generator

`edm_moi_chembl_15_39.pt`

`adj_mat_seer_chembl_15_39.pt`

---

## 🐍 Python API

See interactive examples: `./python_api_demo.ipynb`

```python
from rdkit import Chem
from mlconfgen import MLConformerGenerator, evaluate_samples

model = MLConformerGenerator(
                             edm_weights="./edm_moi_chembl_15_39.pt",
                             adj_mat_seer_weights="./adj_mat_seer_chembl_15_39.pt",
                             diffusion_steps=100,
                            )

reference = Chem.MolFromMolFile('./assets/demo_files/ceyyag.mol')

samples = model.generate_conformers(reference_conformer=reference, n_samples=20, variance=2)

aligned_reference, std_samples = evaluate_samples(reference, samples)
```
---

## 🚀 Overview

This solution employs:

- **Equivariant Diffusion Model (EDM) [[1]](https://doi.org/10.48550/arXiv.2203.17003)**: For generating atom coordinates and types under a shape constraint.
- **Graph Convolutional Network (GCN) [[2]](https://doi.org/10.1039/D3DD00178D)**: For predicting atom adjacency matrices.
- **Deterministic Standardization Pipeline**: For refining and validating generated molecules.

---

## 🧠 Model Training

- Trained on **1.6 million** compounds from the **ChEMBL** database.
- Filtered to molecules with **15–39 heavy atoms**.
- Supported elements: `H, C, N, O, F, P, S, Cl, Br`.

---

## 🧪 Standardization Pipeline

The generated molecules are post-processed through the following steps:

- Largest Fragment picker
- Valence check
- Kekulization
- RDKit sanitization
- Constrained Geometry optimization via **MMFF94** Molecular Dynamics

---

## 📏 Evaluation Pipeline

Aligns and Evaluates shape similarity between generated molecules and a reference using
**Shape Tanimoto Similarity [[3]](https://doi.org/10.1007/978-94-017-1120-3_5 )** via Gaussian Molecular Volume overlap.

> Hydrogens are ignored in both reference and generated samples for this metric.

---

## 📊 Performance (100 Denoising Steps)

*Tested on 100,000 samples using 1,000 CCDC Virtual Screening [[4]](https://www.ccdc.cam.ac.uk/support-and-resources/downloads/) reference compounds.*

- ⏱ **Avg time to generate 50 valid samples**: 11.46 sec (NVIDIA H100)
- ⚡️ **Generation speed**: 4.18 valid molecules/sec
- 💾 **GPU memory (per generation thread)**: Up to 4.0 GB
- 📐 **Avg Shape Tanimoto Similarity**: 53.32%
- 🎯 **Max Shape Tanimoto Similarity**: 99.69%
- 🔬 **Avg Chemical Tanimoto Similarity (2-hop 2048-bit Morgan Fingerprints)**: 10.87%
- 🧬 **% Chemically novel (vs. training set)**: 99.84%
- ✔️ **% Valid molecules (post-standardization)**: 48%
- 🔁 **% Unique molecules in generated set**: 99.94%
- 📎 **Fréchet Fingerprint Distance (2-hop 2048-bit Morgan Fingerprints)**:  
  - To ChEMBL: 4.13  
  - To PubChem: 2.64  
  - To ZINC (250k): 4.95

---

## Generation Examples

![ex1](https://raw.githubusercontent.com/Membrizard/ml_conformer_generator/main/assets/ref_mol/molecule_1.png)
![ex2](https://raw.githubusercontent.com/Membrizard/ml_conformer_generator/main/assets/ref_mol/molecule_2.png)
![ex3](https://raw.githubusercontent.com/Membrizard/ml_conformer_generator/main/assets/ref_mol/molecule_3.png)
![ex4](https://raw.githubusercontent.com/Membrizard/ml_conformer_generator/main/assets/ref_mol/molecule_4.png)

---

## 💾 Access & Licensing

The **Python package and inference code are available on GitHub** under Apache 2.0 License
> https://github.com/Membrizard/ml_conformer_generator

The trained model **Weights** are available at

> https://huggingface.co/Membrizard/ml_conformer_generator

And are licensed under CC BY-NC-ND 4.0

The usage of the trained weights for any profit-generating activity is restricted.

For commercial licensing and inference-as-a-service, contact:
[Denis Sapegin](https://github.com/Membrizard)

---

## ONNX Inference:
For torch Free inference an ONNX version of the model is present. 

Weights of the model in ONNX format are available at:
> https://huggingface.co/Membrizard/ml_conformer_generator

`egnn_chembl_15_39.onnx`

`adj_mat_seer_chembl_15_39.onnx`


```python
from mlconfgen import MLConformerGeneratorONNX
from rdkit import Chem

model = MLConformerGeneratorONNX(
                                 egnn_onnx="./egnn_chembl_15_39.onnx",
                                 adj_mat_seer_onnx="./adj_mat_seer_chembl_15_39.onnx",
                                 diffusion_steps=100,
                                )

reference = Chem.MolFromMolFile('./assets/demo_files/yibfeu.mol')
samples = model.generate_conformers(reference_conformer=reference, n_samples=20, variance=2)

```
Install ONNX GPU runtime (if needed):
`pip install onnxruntime-gpu`

---
## Export to ONNX
An option to compile the model to ONNX is provided

requires `onnxscript==0.2.2`

`pip install onnxscript`

```python
from mlconfgen import MLConformerGenerator
from onnx_export import export_to_onnx

model = MLConformerGenerator()
export_to_onnx(model)
```
This compiles and saves the ONNX files to: `./`

## Streamlit App

![streamlit_app](https://raw.githubusercontent.com/Membrizard/ml_conformer_generator/main/assets/app_ui/streamlit_app.png)

### Running
- Move the trained PyTorch weights into `./streamlit_app`

`./streamlit_app/edm_moi_chembl_15_39.pt`

`./streamlit_app/adj_mat_seer_chembl_15_39.pt`

- Install the dependencies `pip install -r ./streamlit_app/requirements.txt`

- Bring the app UI up:
  ```commandline
  cd ./streamlit_app
  streamlit run app.py
  ```

### Streamlit App Development

1. To enable development mode for the 3D viewer (`stspeck`), set `_RELEASE = False` in `./streamlit/stspeck/__init__.py`.

2. Navigate to the 3D viewer frontend and start the development server:
   ```commandline
   cd ./frontend/speck/frontend
   npm run start
   ```
   
   This will launch the dev server at `http://localhost:3001`

3. In a separate terminal, run the Streamlit app from the root frontend directory: 
   ```commandline
   cd ./streamlit_app
   streamlit run app.py
   ```

4. To build the production version of the 3D viewer, run:
   ```commandline
   cd ./streamlit_app/stspeck/frontend
   npm run build
   ```
