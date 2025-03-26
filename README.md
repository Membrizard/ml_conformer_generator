# ML Conformer Generator (ChemBl)

A tool for shape-constrained molecule generation.

The solution utilises an Equivariant Diffusion Model (EDM) [] to generate atom coordinates and types using a shape constrain,
which are then used by a GCN model [] for atom adjacency prediction. Both models outputs are combined to construct
molecules, which are then passed through a standardisation pipeline.

The EDM and GCN models were trained on a 1.6M compounds having 15-39 heavy atoms selected from the ChEMBL database.
The solution may use the following elements for the molecule generation: H, C, N, O, F, P, S, Cl, Br

The standardiser pipeline uses the following steps:
- Checks for atom valence
- Kekulises molecules
- RDkit Sanitization
- Molecular Dynamics geometry optimisation with MMFF94

The evaluation pipeline assesses the shape similarity of the generated molecules to a reference. 
The assessment is based on a shape tanimoto similarity score [], calculated using Gaussian Molecular Volume intersections.
the shape Tanimoto similarity of a generated molecule to a reference is calculated ignoring hydrogens in both reference and generated sample.

Example performance of the model as evaluated on 100k generated samples

(Used 1000 compounds from ccdc GOLD Virtual Screening dataset for generation)
*1000 Denoising Steps*:

- The average time for generation of 50 valid samples is 96 sec (NVidia H100)
- Average Generation speed (NVidia H100) - 0.5 molecule/sec (valid)
- Estimated GPU memory Consumtion per single Generation thread - 2.5 GB
- Average Shape Tanimoto similarity - 53.38%
- Maximum Shape Tanimoto similarity - 99.21%
- Average Chemical Tanimoto similarity - 10.8%
- % Of chemically unique molecules in reference to training dataset (not found in training dataset) - 99.81%
- % Of valid molecules in generated batch (as defined by the standardisation pipeline) - 48.59%
- % Of chemically unique molecules within the generated set (as evaluated on 80k generated molecules) - 99.80%
- Average Generation speed (NVidia H100) - 0.5 molecule/sec (valid)
- Freschet Fingerprint Distance (2048) [] to ChEMBL - 3.98 to PubChem - 2.57 to ZINC (250k drugs) - 5.38

*100 Denoising Steps*:

- The average time for generation of 50 valid samples is 96 sec (NVidia H100)
- Average Generation speed (NVidia H100) - 0.5 molecule/sec (valid)
- Estimated GPU memory Consumtion per single Generation thread - 2.5 GB
- Average Shape Tanimoto similarity - 53.38%
- Maximum Shape Tanimoto similarity - 99.21%
- Average Chemical Tanimoto similarity - 10.8%
- % Of chemically unique molecules in reference to training dataset (not found in training dataset) - 99.81%
- % Of valid molecules in generated batch (as defined by the standardisation pipeline) - 48.59%
- % Of chemically unique molecules within the generated set (as evaluated on 80k generated molecules) - 99.80%
- Average Generation speed (NVidia H100) - 0.5 molecule/sec (valid)
- Freschet Fingerprint Distance (2048) [] to ChEMBL - 3.98 to PubChem - 2.57 to ZINC (250k drugs) - 5.38


Generator requirements are in  ./ml_conformer_generator/generator_requirements.txt

Frontend requirements are in ./frontend/fronted_requirements.txt


## Usage

### Python API
Look for interactive example in `./python_api_demo.ipynb`

```
from rdkit import Chem
from ml_conformer_generator import MLConformerGenerator, evaluate_samples

model = MLConformerGenerator(device="cpu")

reference = Сhem.MolFromMolFile('')

samples = model.generate_conformers(reference_conformer=reference, n_samples=20)
    
aligned_reference, std_samples = evaluate_samples(reference, samples)

```

### API Server
- Run `docker compose up -d --build`
- The api server should be available at http:/0.0.0.0:8000
- The Swagger documentation is available at http:/0.0.0.0:8000/docs
- Generation endpoint http:/0.0.0.0:8000/generate

#### Request Schema
```
{
  "reference_mol": {
    "type": "string",
    "content": "string"
  },
  "n_samples": 0,
  "variance": 0
}
```
#### Response Schema
```
{
  "results": {
    "aligned_reference": "string",
    "generated_molecules": [
        {
            "mol_block": "string",
            "shape_tanimoto": 0.1,
            "chemical_tanimoto": 0.1,
            "svg": "string"
        }
    ]
  },
  "errors": "string"
}

```

### Frontend 

#### Running
- To bring the app UI up:
```
cd ./frontend
streamlit run app_ui.py
```


#### Development
- To switch 3D viewer (stspeck) to development set `_RELEASE=False` in `./frontend/stspeck/__init__.py`
- Go to ./frontend/speck/fronted and run `npm run start` after that dev speck will run on http://localhost:3001
- After that run streamlit app from ./frontend
```
cd ./frontend
streamlit run app_ui.py
```
- To build the 3D viewer go to ./frontend/speck/fronted and run `npm start build`
