from typing import List, Tuple

import torch
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from .config import DIMENSION
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
    :param positions: coordinates tensor
    :param one_hot: one-hot encoded atom types
    :param node_mask: node mask
    :param atom_decoder: atom decoder dictionary
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
        if mol is not None:
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
    mols: List[Chem.Mol], dimension: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Chem.Mol]]:
    """
    Prepares input for AdjMatSeer Model
    :param mols: a list of raw molecules as rdkit Mol objects
    :param dimension: dimension of the input - int
    :param device: a device to prepare input on torch.device
    """
    canonicalised_samples = []

    n_samples = len(mols)

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
    """
    Redefines bonds in a given molecule according to an adjacency matrix:
    :param mol: rdkit Mol object
    :param adj_mat: adjacency matrix as torch.Tensor
    :return: molecule with redefined bonds as rdkit Mol
    """
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


def prepare_masks(
    n_nodes: torch.Tensor,
    max_n_nodes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare node and edge masks
    :param n_nodes: a list of sizes of requested molecules (n_samples, 1)
    :param max_n_nodes: maximal number of nodes
    :param device: device to prepare masks on torch. device
    :return: node_mask, edge_mask
    """

    batch_size = n_nodes.size(0)

    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0 : n_nodes[i]] = 1

    # Compute edge_mask
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    return node_mask, edge_mask


def prepare_edm_input(
    n_samples: int,
    reference_context: torch.Tensor,
    context_norms: dict,
    min_n_nodes: int,
    max_n_nodes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares Input for EDM model
    :param n_samples: number of molecules to generate
    :param reference_context: context to use for generation
    :param context_norms: Values for normalisation of context
    :param min_n_nodes: minimal allowable molecule size
    :param max_n_nodes: maximal allowable molecule size
    :param device: device to prepare input for - torch.device
    :return: a tuple of tensors ready to be used by the EDM
    """
    # Create a random list of sizes between min_n_nodes and max_n_nodes of length n_samples

    nodesxsample = torch.randint(min_n_nodes, max_n_nodes + 1, (n_samples,))

    node_mask, edge_mask = prepare_masks(
        n_nodes=nodesxsample,
        max_n_nodes=max_n_nodes,
        device=device,
    )

    normed_context = (
        (reference_context - context_norms["mean"]) / context_norms["mad"]
    ).to(device)

    batch_context = normed_context.unsqueeze(0).repeat(n_samples, 1)

    batch_context = batch_context.unsqueeze(1).repeat(1, max_n_nodes, 1) * node_mask

    return (
        node_mask,
        edge_mask,
        batch_context,
    )


def prepare_fragment(
    n_samples: int,
    fixed_fragment: Chem.Mol,
    device: torch.device,
    max_n_nodes: int = DIMENSION,
    min_n_nodes: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepares Fixed Fragment for Inpainting. Converts Mol to latent Z tensor, ready for injection
    :param n_samples: required batch size of the prepared latent fragment - number of molecules to generate
    :param fixed_fragment: fragment to prepare - rdkit Mol
    :param device: device to prepare input for - torch.device
    :param max_n_nodes: possible maximum number of nodes - for padding - int
    :param min_n_nodes: possible minimum number of nodes - int
    :return: Latent representation of the fragment and a mask,
             indicating which atoms in the latent representation are fixed
    """

    coord, h = ifm_get_xh_from_fragment(fixed_fragment, device)
    n_atoms = coord.size(0)

    # Check that fragment size is adequate
    if n_atoms >= min_n_nodes:
        raise ValueError(
            "Fragment must contain fewer atoms than minimum generation size."
        )
    if n_atoms >= max_n_nodes:
        raise ValueError(
            "Fragment has more atoms than the maximum number of atoms requested."
        )

    x = torch.nn.functional.pad(coord, (0, 0, 0, max_n_nodes - n_atoms), "constant", 0)
    h = torch.nn.functional.pad(h, (0, 0, 0, max_n_nodes - n_atoms), "constant", 0)

    # Batch x and h
    x = x.repeat(n_samples, 1, 1)
    h = h.repeat(n_samples, 1, 1)
    z_known = torch.cat([x, h], dim=2).to(device)

    fixed_mask = torch.zeros(
        (n_samples, max_n_nodes, 1), dtype=torch.float32, device=device
    )
    fixed_mask[:, :n_atoms, 0] = 1.0

    return z_known, fixed_mask


def ifm_get_xh_from_fragment(
    fixed_fragment: Chem.Mol, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get coordinates and atom types as tensors for a fragment
    :param fixed_fragment: fragment as rdkit Mol object
    :param device: device to prepare tensors on torch.device
    :return: fixed_fragment_x, fixed_fragment_h
    """
    # Get coordinates of the fixed fragment
    ff_mol = Chem.RemoveAllHs(fixed_fragment)
    ff_conformer = ff_mol.GetConformer()
    ff_x = torch.tensor(ff_conformer.GetPositions(), dtype=torch.float32)

    # Get atom types of a fixed fragment
    ff_structure = MolGraph.from_mol(mol=ff_mol, remove_hs=True)
    ff_n_atoms = ff_x.size(0)

    ff_h = ff_structure.one_hot_elements_encoding(
        ff_n_atoms
    )  # Atom types of a fixed fragment

    ff_x = ff_x.to(device)
    ff_h = ff_h.to(device)

    return ff_x, ff_h


def ifm_prepare_gen_fragment_context(
    fixed_fragment_x: torch.Tensor,
    reference_context: torch.Tensor,
    context_norms: dict,
    n_nodes: torch.Tensor,
    max_n_nodes: int,
    min_n_nodes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare Contexts for Generation of Individual Fragments based on Reference and Fixed Fragment MOI tensors.
    Fixed Fragment should be expressed in the same coordinate system as the reference, The origin of such coordinate
    system should be place in the COM of the Reference.

    :param fixed_fragment_x:  coordinates of the fixed fragment atoms
    :param reference_context: Principal components of MOI Tensor of the reference shape (3)
    :param context_norms: Values to norm the context for individual fragment generation
    :param n_nodes: a list of sizes of requested molecules (n_samples, 1)
    :param max_n_nodes: a maximal number of nodes allowed for the whole generated molecule
    :param min_n_nodes: a minimal number of nodes allowed for the whole generated molecule
    :param device: device to prepare input for - torch.device
    :return: frag_node_mask, frag_edge_mask, batched_normed_frag_context, shift, rotation
    """
    batch_size = n_nodes.size(0)
    ff_n_atoms = fixed_fragment_x.size(0)

    # Check that fragment size is adequate
    if ff_n_atoms >= min_n_nodes:
        raise ValueError(
            "Fragment must contain fewer atoms than minimum generation size."
        )
    if ff_n_atoms >= max_n_nodes:
        raise ValueError(
            "Fragment has more atoms than the maximum number of atoms requested."
        )

    masses_ff = torch.ones(ff_n_atoms, device=device)

    # Fixed fragment MOI around origin
    moi_ff = get_moment_of_inertia_tensor(fixed_fragment_x, masses_ff)  # (3, 3)

    # Reference MOI as diagonal matrix (expand to B)
    moi_ref = torch.diag(reference_context)  # (3, 3)

    moi_ref_batch = moi_ref.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 3, 3)
    moi_ff_batch = moi_ff.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 3, 3)

    # Gen frag MOI around origin
    moi_gen_origin = moi_ref_batch - moi_ff_batch  # (B, 3, 3)

    # COM of fixed fragment
    com_ff = fixed_fragment_x.mean(dim=0)  # (3,)

    gen_n_atoms = n_nodes.view(batch_size, 1).float() - ff_n_atoms  # (B, 1)

    # COM of generated fragments in respect to number of atoms in each
    shift = (ff_n_atoms * com_ff.view(1, 3)) / gen_n_atoms  # (B, 3)

    # Shift MOI to COM of generated fragment
    moi_gen_com = shift_moi_to_com_batch(
        moi_gen_origin, shift, gen_n_atoms
    )  # (B, 3, 3)

    # Diagonalize MOI of generated fragment
    frag_context, rotation = torch.linalg.eigh(moi_gen_com)  # (B, 3), (B, 3, 3)
    normed_frag_context = (
        (frag_context - context_norms["mean"]) / context_norms["mad"]
    ).to(device)

    max_n_nodes_frag = max_n_nodes - ff_n_atoms

    frag_node_mask, frag_edge_mask = prepare_masks(
        n_nodes=gen_n_atoms.long(),
        max_n_nodes=max_n_nodes_frag,
        device=device,
    )

    batched_normed_frag_context = (
        normed_frag_context.unsqueeze(1).repeat(1, max_n_nodes_frag, 1) * frag_node_mask
    )

    rotation = rotation.to(device)
    shift = shift.to(device)

    return frag_node_mask, frag_edge_mask, batched_normed_frag_context, shift, rotation


def ifm_prepare_fragments_for_merge(
    fixed_fragment_x: torch.Tensor,
    fixed_fragment_h: torch.Tensor,
    gen_fragments_x: torch.Tensor,
    gen_fragments_h: torch.Tensor,
    device: torch.device,
    max_n_nodes: int,
):
    """
    Prepares Multiple Fragments for Merge. Prepares latent Z tensor, ready for injection and mask for a fixed fragment.

    :param fixed_fragment_x: Coordinates of a fixed fragment as torch.Tensor
    :param fixed_fragment_h: One-hot encoded atom types of a fixed fragment as torch.Tensor
    :param gen_fragments_x: Batch of coordinates of generated fragments - torch.Tensor
    :param gen_fragments_h: Batch of One-hot encoded atom types of generated fragments - torch.Tensor
    :param device: device to prepare output on torch.device
    :param max_n_nodes: maximal allowable number of atoms
    :return: z_known, fixed_mask, n_samples
    """

    n_samples = gen_fragments_x.size(0)

    ff_n_atoms = fixed_fragment_x.size(0)

    # Add a batch dimension
    ff_x = fixed_fragment_x.unsqueeze(0)  # Shape: (1, N, 3)
    ff_h = fixed_fragment_h.unsqueeze(0)  # Shape: (1, N, F)

    # Repeat across batch dimension
    ff_x_batched = ff_x.repeat(n_samples, 1, 1).to(device)  # Shape: (n_samples, N, 3)
    ff_h_batched = ff_h.repeat(n_samples, 1, 1).to(device)

    x_prep = torch.cat([ff_x_batched, gen_fragments_x], dim=1)
    h_prep = torch.cat([ff_h_batched, gen_fragments_h], dim=1)

    z_known = torch.cat([x_prep, h_prep], dim=2)

    # The fixed fragment is always in the first place - so we set fixed mask to have 1s only
    # on the first ff_n_atoms elements of z_known

    fixed_mask = torch.zeros(
        (n_samples, max_n_nodes, 1), dtype=torch.float32, device=device
    )
    fixed_mask[:, :ff_n_atoms, 0] = 1.0

    return z_known, fixed_mask


def inverse_coord_transform(
    coord: torch.Tensor, shift: torch.Tensor, rotation: torch.Tensor
) -> torch.Tensor:
    """
    Inverse shift and Rotation transformation to a batch of xyz coordinates sets
    :param coord: Batch of Coordinates to be modified (batch_size, N, 3)
    :param shift: Batch of shifts to be applied (batch_size, 3)
    :param rotation: Batch of Rotations to be applied (batch_size, 3, 3)
    :return: Modified Coordinates
    """
    # Rotate first
    batch_size = coord.size(0)
    x_rotated = torch.bmm(coord, torch.transpose(rotation, 1, 2))
    # Translate second
    x_translated = x_rotated - shift.view(batch_size, 1, 3)

    return x_translated


def shift_moi_to_com_batch(
    moi_origin: torch.Tensor, r_coms: torch.Tensor, masses: torch.Tensor
) -> torch.Tensor:
    """
    Translates moment of inertia from the origin to multiple guessed centers of mass
    using the inverse parallel axis theorem.

    :param moi_origin: (3, 3) Inertia tensor around origin (shared across batch)
    :param r_coms: (B, 3) Vectors from origin to guessed COMs
    :param masses: (B,) Total masses per example
    :return: I_coms: (B, 3, 3) Inertia tensors about the guessed COMs
    """
    batch_size = r_coms.size(0)
    i_3 = torch.eye(3, device=r_coms.device).expand(batch_size, 3, 3)  # (B, 3, 3)

    r = r_coms.view(batch_size, 3, 1)  # (B, 3, 1)
    r_outer = r @ r.transpose(1, 2)  # (B, 3, 3)
    r_norm_sq = (r_coms**2).sum(dim=1).view(batch_size, 1, 1)  # (B, 1, 1)

    masses = masses.view(batch_size, 1, 1)  # (B, 1, 1)
    shift = masses * (r_norm_sq * i_3 - r_outer)  # (B, 3, 3)
    shift = shift.to(moi_origin.device)

    return moi_origin - shift  # (B, 3, 3)
