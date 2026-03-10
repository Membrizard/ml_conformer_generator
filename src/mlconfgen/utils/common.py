"""
Shared constants and pure-RDKit functions used by both PyTorch and ONNX code paths.
"""

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

from .config import PERMITTED_ELEMENTS

elements_decoder = {x: i for i, x in enumerate(sorted(PERMITTED_ELEMENTS))}

# allowable node and edge features
allowable_features = {
    "possible_atomic_num_list": list(range(1, 35)),
    "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
}

elements_dict = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
}

bond_type_dict = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC,
}


def canonicalise(mol: Chem.Mol) -> Chem.Mol:
    """
    Bring order of atoms in the molecule to canonical based on generic one-order connectivity
    :param mol: Mol object with unordered atoms
    :return: Mol object with canonicalised order of atoms
    """
    # Guess simple 1-order connectivity and re-order the molecule
    rdDetermineBonds.DetermineConnectivity(mol)
    _ = Chem.MolToSmiles(mol)
    order_str = mol.GetProp("_smilesAtomOutputOrder")

    order_str = order_str.replace("[", "").replace("]", "")
    order = [int(x) for x in order_str.split(",") if x != ""]

    mol_ordered = Chem.RenumberAtoms(mol, order)

    return mol_ordered


def apply_transform(coord, shift, rotation):
    """
    Apply Translation -> Rotation transform to a set of coordinates
    :param coord: Coordinates
    :param shift: Shift (Translation) matrix
    :param rotation: Rotation matrix
    :returns: Transformed coordinates
    """
    coord_shifted = coord + shift
    coord_transformed = coord_shifted @ rotation
    return coord_transformed


def set_conformer_positions(mol, coord):
    conf = mol.GetConformer()
    for i, point in enumerate(coord):
        x, y, z = point.tolist()
        conf.SetAtomPosition(i, Point3D(x, y, z))

    return mol
