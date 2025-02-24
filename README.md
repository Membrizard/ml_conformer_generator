# ML Conformer Generator (ChemBl)

A tool to generate random molecules, which have a conformer similar in shape to a reference.

The solution utilises an Equivariant Diffusion Model (EDM) [] to generate atom coordinates and types using a shape constrain,
which are then used by a GCN model [] for atom adjacency prediction. Both models outputs are combined to construct
molecules, which are then passed through a standardisation pipeline.

The EDM and GCN models were trained on a 1.6M compounds having 15-39 heavy atoms selected from the ChEMBL database.
The solution may use the following elements for the molecule generation: H, C, N, O, F, P, S, Cl, Br

The standardiser pipeline uses the following steps:
- Checks for atom valence
- Kekulises molecules
- RDkit Sanitization
- Molecular Dynamics geometry optimisation with MMFF

The evaluation pipeline assesses the shape similarity of the generated molecules to a reference. 
The assesment is based on a shape tanimoto similarity score [], calculated using Gaussian Molecular Volume intersections.
the shape Tanimoto similarity of a generated molecule to a reference is calculated ignoring hydrogens in both reference and generated sample.

Example performance of the model as evaluated on 100k samples
- The estimated average time for generation of 50 valid samples is 90-160 sec (GPU)
- Average Shape Tanimoto similarity - 49.95%
- % Of unique molecules (not found in training dataset) - 99.85%
- % of valid molecules in generated batch (as defined by the stadardisation pipeline) - 51.32%

Generator requirements are in  ./ml_conformer_generator/generator_requirements.txt

Frontend requirements are in ./frontend/fronted_requirements.txt


## Usage

### Python Interface
Look for interactive example in `./ml_conformer_generator_app_demo.ipynb`

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

