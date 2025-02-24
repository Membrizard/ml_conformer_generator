#### ML Conformer Generator (ChemBl)

A tool to generate random molecules, which have a conformer similar in shape to a reference.

The solution utilises an Equivariant Diffusion Model (EDM) to generate atom coordinates and types using a shape constrain,
which are then used by a GCN model [] for atom adjacency prediction. Both models outputs are combined to construct
molecules, which are then passed through a standardisation pipeline.

The EDM and GCN models were trained on a 1.6M compounds having 15-39 heavy atoms selected from the ChEMBL database.
The solution may use the following elements for the molecule generation: H, C, N, O, F, P, S, Cl, Br

The standardiser pipeline uses the following steps:
- Checks for atom valence and kekulisation
- ChemBL standardisation
- Molecular Dynamics geometry optimisation with MMFF

The evaluation pipeline assesses the shape similarity of the generated molecules to a reference. 
The assesment is based on a shape tanimoto similarity score [], calculated using Gaussian Molecular Volume intersections.
the shape Tanimoto similarity of a generated molecule to a reference is calculated ignoring hydrogens in both reference and generated sample.

Example performance of the model as evaluated on 100k samples
- The estimated average time for generation of 50 valid samples is 90 sec (GPU)
- Average Tanimoto similarity - 
- % Of molecules not in training dataset -
- % of valid molecules in generated batch (as defined by the stadardisation pipeline) - 

Generator requirements are in  ./ml_conformer_generator/generator/requirements.txt
Frontend requirements are in ./frontend/fronted_requirements.txt


#### Usage

Python Interface
```
from rdkit import Chem
from ml_conformer_generator import MLConformerGenerator
from cheminformatics import evaluate_samples



model = MLConformerGenerator(device="cpu")

reference = Сhem.MolFromMolFile('')

samples = model.generate_conformers(reference_conformer=reference, n_samples=20)
    
_, std_samples = evaluate_samples(reference, samples)



```

CMD interface

Docker / HTTP

#### Frontend Shit

To start in dev, go to frontend/speck/fronted and run `npm run start`
after that dev speck will run on http://localhost:3001

After that run streamlit app from ./frontend 
```
cd ./frontend
streamlit run app_ui.py
```

3D viewer is now built

just run:
```
cd ./frontend
streamlit run app_ui.py
```
to start the app
