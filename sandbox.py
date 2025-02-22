import rdkit.Chem
from rdkit import Chem
from rdkit.Chem import rdDistGeom
import json


for i in range(6):
    with open(f"./generation_examples/generation_example_{i+1}.json") as json_file:
        data = json.load(json_file)
