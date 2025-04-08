import random
from typing import List, Tuple

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
    one_hot: torch.Tensor,
    node_mask: torch.Tensor = None,
    atom_decoder: dict = None,
) -> List[Chem.Mol]:
    """
    Convert EDM Samples to RDKit mol objects
    :param positions:
    :param one_hot:
    :param node_mask:
    :param atom_decoder:
    :return: a list of samples as RDKit Mol objects without bond information
    """
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


def get_moment_of_inertia_tensor(
    coord: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """
    Calculate a Moment of Inertia tensor
    :return: Moment of Inertia Tensor in input coordinates
    """
    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]

    # Diagonal elements
    i_xx = torch.sum(weights * (y**2 + z**2))
    i_yy = torch.sum(weights * (x**2 + z**2))
    i_zz = torch.sum(weights * (x**2 + y**2))

    # Off-diagonal elements
    i_xy = -torch.sum(x * y)
    i_xz = -torch.sum(x * z)
    i_yz = -torch.sum(y * z)

    # Construct the MOI tensor
    moi_tensor = torch.tensor(
        [[i_xx, i_xy, i_xz], [i_xy, i_yy, i_yz], [i_xz, i_yz, i_zz]],
        dtype=torch.float32,
    )

    return moi_tensor


def get_context_shape(coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Finds the principal axes for the conformer,
    and calculates Moment of Inertia tensor for the conformer in principal axes.
    All atom masses are considered equal to one, to capture shape only.
    :param coord: initial coordinates of the atoms
    :return: Principal components of MOI tensor, and coordinates rotated to a principal frame as a tuple of tensors
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


def prepare_adj_mat_seer_input(
    mols: List[Chem.Mol], n_samples: int, dimension: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Chem.Mol]]:
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


def redefine_bonds(mol: Chem.Mol, adj_mat: torch.Tensor) -> Chem.Mol:
    n = mol.GetNumAtoms()
    # Pass the molecule through xyz block to remove bonds and all extra atom properties
    i_xyz = Chem.MolToXYZBlock(mol)
    c_mol = Chem.MolFromXYZBlock(i_xyz)
    ed_mol = Chem.EditableMol(c_mol)

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


def prepare_edm_input(
    n_samples: int,
    reference_context: torch.Tensor,
    context_norms: dict,
    min_n_nodes: int,
    max_n_nodes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Create a random list of sizes between min_n_nodes and max_n_nodes of length n_samples
    nodesxsample = []

    for n in range(n_samples):
        nodesxsample.append(random.randint(min_n_nodes, max_n_nodes))

    nodesxsample = torch.tensor(nodesxsample)

    batch_size = nodesxsample.size(0)

    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0 : nodesxsample[i]] = 1

    # Compute edge_mask

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    normed_context = (
        (reference_context - context_norms["mean"]) / context_norms["mad"]
    ).to(device)

    batch_context = normed_context.unsqueeze(0).repeat(batch_size, 1)

    batch_context = batch_context.unsqueeze(1).repeat(1, max_n_nodes, 1) * node_mask

    return (
        node_mask,
        edge_mask,
        batch_context,
    )
