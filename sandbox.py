import time

from rdkit import Chem
from rdkit.Chem import Draw, rdDistGeom

from ml_conformer_generator import MLConformerGeneratorONNX, evaluate_samples

device = "cpu"
generator = MLConformerGeneratorONNX(
    egnn_onnx="egnn_moi_chembl_15_39.onnx",
    adj_mat_seer_onnx="adj_mat_seer_chembl_15_39.onnx",
    diffusion_steps=100,
)

ref_mol = Chem.MolFromSmiles("Cc1cccc(N2CCC[C@H]2c2ccncn2)n1")
rdDistGeom.EmbedMolecule(ref_mol, forceTol=0.001, randomSeed=12)

# Generate Samples
start = time.time()
samples = generator.generate_conformers(
    reference_conformer=ref_mol, n_samples=4, variance=2
)
print(f"generation complete in {round(time.time() - start, 2)}")

# # Characterise samples
# _, std_samples = evaluate_samples(ref_mol, samples)
#
# mols = []
# for sample in std_samples:
#     mol = Chem.MolFromMolBlock(sample['mol_block'])
#     mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
#     mol.SetProp("Shape_Tanimoto", str(sample['shape_tanimoto']))
#     mols.append(mol)
#
# Draw.MolsToGridImage(mols)
