from rdkit import Chem


def prepare_speck_model(mol: Chem.Mol, reference: Chem.Mol = None):
    conformer = mol.GetConformer(-1)
    mol_json = {"atoms": [], "bonds": []}

    for i, atom in enumerate(mol.GetAtoms()):
        position = conformer.GetAtomPosition(i)
        mol_json["atoms"].append(
            {
                "symbol": atom.GetSymbol(),
                "x": position.x,
                "y": position.y,
                "z": position.z,
            }
        )

    for bond in mol.GetBonds():
        mol_json["bonds"].append(
            {"begin_atom": bond.GetBeginAtomIdx(), "end_atom": bond.GetEndAtomIdx()}
        )

    if reference:
        last_atom_idx = len(mol_json["atoms"])
        ref_conformer = reference.GetConformer(-1)
        for j, atom in enumerate(reference.GetAtoms()):
            position = ref_conformer.GetAtomPosition(j)
            mol_json["atoms"].append(
                {"symbol": "Ref", "x": position.x, "y": position.y, "z": position.z}
            )

        for bond in reference.GetBonds():
            mol_json["bonds"].append(
                {
                    "begin_atom": bond.GetBeginAtomIdx() + last_atom_idx,
                    "end_atom": bond.GetEndAtomIdx() + last_atom_idx,
                }
            )

    return mol_json
