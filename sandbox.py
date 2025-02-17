import rdkit.Chem
from rdkit import Chem
from rdkit.Chem import rdDistGeom


mol = Chem.MolFromSmiles('C1=CC(=CC=C1C(=O)O)N')
mol = Chem.AddHs(mol)
rdDistGeom.EmbedMolecule(mol, forceTol=0.001, randomSeed=12)
conformer = mol.GetConformer()


def jsonify_mol(mol: rdkit.Chem.Mol):

    conformer = mol.GetConformer(-1)
    mol_json = {'atoms': [], 'bonds': []}

    for i, atom in enumerate(mol.GetAtoms()):
        position = conformer.GetAtomPosition(i)
        mol_json['atoms'].append({'symbol': atom.GetSymbol(), 'x': position.x, 'y': position.y, 'z': position.z} )

    for i, bond in enumerate(mol.GetBonds()):
        mol_json['bonds'].append({'begin_atom': bond.GetBeginAtomIdx(), 'end_atom': bond.GetEndAtomIdx()})

    return mol_json


print(mol_json)


