from rdkit import Chem
from tqdm import tqdm

supplier = Chem.SDMolSupplier("./data/ifm_generation_test_6_15.sdf")
writer = Chem.SDWriter("./data/smoke_test_ifm_generation_5_molecules.sdf")

l_supp = [mol for mol in supplier]

names = []

for mol in tqdm(l_supp[:5]):
    writer.write(mol)


