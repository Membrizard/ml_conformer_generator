#### ML Conformer Generator (ChemBl)

A tool to generate random molecules, which have a conformer similar in shape to a reference.

The solution utilises an Equivariant Diffusion Model (EDM) to generate atom coordinates and types using a shape constrain,
which are then used by a GCN model [] for atom adjacency prediction. Both models outputs are combined to construct
molecules, which are then passed through a standardisation pipeline.

The EDM and GCN models were trained on a 1.6M compounds having 15-39 heavy atoms selected from the ChEMBL database.
The solution may use the following elements for the molecule generation: H, C, N, O, F, P, S, Cl, Br

The standardiser pipeline uses the following steps for 

The evaluation pipeline assesses the shape similarity of the generated molecules to a reference. 
The assesment is based on a shape tanimoto similarity score [], calculated based on intersection
of Gaussian Molecular volumes of the reference and candidate molecules.



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
