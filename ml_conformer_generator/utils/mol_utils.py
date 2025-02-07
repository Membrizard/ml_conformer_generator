import torch
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from .molgraph import MolGraph

bond_type_dict = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC,
}


def samples_to_rdkit_mol(
    positions,
    one_hot,
    node_mask=None,
    atom_decoder: dict = None,
):
    rdkit_mols = []

    if node_mask is not None:
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [one_hot.size(1)] * one_hot.size(0)

    for batch_i in range(one_hot.size(0)):
        xyz_block = "%d\n\n" % atomsxmol[batch_i]
        atoms = torch.argmax(one_hot[batch_i], dim=1)
        n_atoms = int(atomsxmol[batch_i])
        for atom_i in range(n_atoms):
            atom = atoms[atom_i]
            atom = atom_decoder[atom.item()]
            xyz_block += "%s %.9f %.9f %.9f\n" % (
                atom,
                positions[batch_i, atom_i, 0],
                positions[batch_i, atom_i, 1],
                positions[batch_i, atom_i, 2],
            )

        mol = Chem.MolFromXYZBlock(xyz_block)
        rdkit_mols.append(mol)

    return rdkit_mols


def get_moment_of_inertia_tensor(coord: torch.Tensor, weights: torch.Tensor):
    """
    Calculate a Moment of Inertia tensor
    :return: Moment of Inertia Tensor in input coordinates
    """
    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]

    # Diagonal elements
    Ixx = torch.sum(weights * (y**2 + z**2))
    Iyy = torch.sum(weights * (x**2 + z**2))
    Izz = torch.sum(weights * (x**2 + y**2))

    # Off-diagonal elements
    Ixy = -torch.sum(x * y)
    Ixz = -torch.sum(x * z)
    Iyz = -torch.sum(y * z)

    # Construct the MOI tensor
    moi_tensor = torch.tensor(
        [[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]], dtype=torch.float32
    )

    return moi_tensor


def get_context_shape(coord):
    """
    Finds the principal axes for the conformer,
    and calculates Moment of Inertia tensor for the conformer in principal axes.
    All atom masses are considered qual to one, to capture shape only.
    :param coord: initial coordinates of the atoms
    :return:
    """
    masses = torch.ones(coord.size(0))
    moi_tensor = get_moment_of_inertia_tensor(coord, masses)
    # Diagonalize the MOI tensor using eigen decomposition
    _, eigenvectors = torch.linalg.eigh(moi_tensor)

    # Rotate points to principal axes
    rotated_points = torch.matmul(coord.to(torch.float32), eigenvectors)

    # Get the three main moments of inertia from the main diagonal
    context = torch.diag(get_moment_of_inertia_tensor(rotated_points, masses))

    return context, rotated_points


def canonicalise(mol):
    # Guess simple 1-order connectivity and re-order the molecule
    rdDetermineBonds.DetermineConnectivity(mol)
    _ = Chem.MolToSmiles(mol)
    order_str = mol.GetProp("_smilesAtomOutputOrder")

    order_str = order_str.replace("[", "").replace("]", "")
    order = [int(x) for x in order_str.split(",") if x != ""]

    mol_ordered = Chem.RenumberAtoms(mol, order)

    return mol_ordered


def distance_matrix(coordinates: torch.Tensor) -> torch.Tensor:
    """
    Generates a distance matrices from a xyz coordinates tensor
    :param coordinates: xyz coordinates tensor
    :return: distance matrix
    """
    n = coordinates.size(0)
    i_mat = coordinates.unsqueeze(1).repeat(
        1, n, 1
    )  # Repeat coordinates tensor along new dimension
    j_mat = i_mat.transpose(0, 1)

    dist_matrix = torch.sqrt(torch.sum(torch.pow(i_mat - j_mat, 2), 2))

    return dist_matrix


def prepare_adj_mat_seer_input(mols, n_samples, dimension, device):
    canonicalised_samples = []

    elements_batch = torch.zeros(n_samples, dimension, dtype=torch.long, device=device)
    dist_mat_batch = torch.zeros(n_samples, dimension, dimension, device=device)
    adj_mat_batch = torch.zeros(n_samples, dimension, dimension, device=device)

    for i, sample in enumerate(mols):
        mol = canonicalise(sample)

        conf = mol.GetConformer()
        coord = torch.tensor(conf.GetPositions())

        structure = MolGraph.from_mol(mol=mol, remove_hs=False)
        elements = structure.elements_vector()
        n_atoms = torch.count_nonzero(elements)

        target_adjacency_matrix = structure.adjacency_matrix()

        sc_adj_mat = torch.argmax(target_adjacency_matrix, dim=2).float() + torch.eye(
            dimension
        )

        sc_adj_mat[sc_adj_mat > 0] = 1

        dist_mat = distance_matrix(coord)
        pad_dist_mat_sc = torch.nn.functional.pad(
            dist_mat,
            (0, dimension - n_atoms, 0, dimension - n_atoms),
            "constant",
            0,
        ) + torch.eye(dimension)

        elements_batch[i] = elements.to(torch.long)
        dist_mat_batch[i] = pad_dist_mat_sc
        adj_mat_batch[i] = sc_adj_mat
        canonicalised_samples.append(mol)

    return elements_batch, dist_mat_batch, adj_mat_batch, canonicalised_samples


def redefine_bonds(mol, adj_mat):
    n = mol.GetNumAtoms()
    ed_mol = Chem.EditableMol(mol)

    # Remove existing bonds

    for bond in mol.GetBonds():
        ed_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    repr_m = torch.tril(torch.argmax(adj_mat, dim=2))
    repr_m = repr_m * (1 - torch.eye(repr_m.size(0), repr_m.size(0)))

    for i in range(n):
        for j in range(n):
            # Find out the bond type by indexing 1 in the matrix bond
            bond_type = repr_m[i, j].item()

            if bond_type != 0:
                ed_mol.AddBond(i, j, bond_type_dict[bond_type])

    new_mol = ed_mol.GetMol()

    return new_mol