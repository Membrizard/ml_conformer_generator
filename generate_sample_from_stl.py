import trimesh
from ml_conformer_generator import MLConformerGenerator, evaluate_samples
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from rdkit.Chem import Draw
import torch

mymesh = trimesh.load("./data/6q8k_pocket_a.stl")
mymesh.density = 0.02
check = mymesh.is_watertight
print(check)
ref_context = torch.tensor(mymesh.principal_inertia_components, dtype=torch.float32)

device = "cuda"
generator = MLConformerGenerator(device=device)


# Generate Samples
samples = generator.generate_conformers(
    reference_context=ref_context, n_atoms=37, n_samples=500, variance=2
)

# Characterise samples
# _, std_samples = evaluate_samples(ref_mol, samples)

writer = Chem.SDWriter("./6q8k_generated_molecules.sdf")
count = 0
for sample in samples:
    if sample.GetNumHeavyAtoms() >= 10:
        count += 1
        writer.write(sample)

print(f"Molecules generated {count}")
