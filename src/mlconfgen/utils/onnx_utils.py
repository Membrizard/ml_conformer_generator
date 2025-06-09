from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdmolops

from .config import DIMENSION, NUM_BOND_TYPES, PERMITTED_ELEMENTS

elements_decoder = {x: i for i, x in enumerate(sorted(PERMITTED_ELEMENTS))}

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

bond_type_dict = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC,
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


class MolGraphONNX:
    """
    A class to handle molecular graphs without PyTorch:
    """

    def __init__(self, x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    @classmethod
    def from_mol(cls, mol: Chem.Mol, remove_hs: bool = True) -> "MolGraphONNX":
        """
        Converts rdkit mol object to MolGraph object
        geometric package. Strips hydrogens from the Mol object. Ignores Atoms and Bonds Chirality,
        Bonds are represented in edge_attrs as integers:
        1 - Single
        2 - Double
        3 - Triple
        4 - Aromatic
        :param mol: rdkit mol object
        :param remove_hs: if H atoms are to be removed
        :return: graph data object with the attributes: x, edge_index, edge_attr
        """
        # Remove hydrogens from the molecule - to simplify graph structure. Ids of atoms remain unchanged.
        if remove_hs:
            mol = rdmolops.RemoveHs(mol)

        out = [0] * len(mol.GetAtoms())
        for atom in mol.GetAtoms():
            element = atom.GetAtomicNum()
            index = atom.GetIdx()
            out[index] = element

        x = np.array(out, dtype=np.float32)

        # bonds
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = (
                    allowable_features["possible_bonds"].index(bond.GetBondType()) + 1
                )
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype=np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype=np.float32)

        else:  # mol has no bonds
            raise ValueError(
                f"Bonds must be specified for the molecule - {mol.GetProp('_Name')}."
            )

        return cls(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def adjacency_matrix(self, padded: bool = True) -> np.ndarray:
        """
        Creates a 0-1 normalised adjacency matrix with a specified size from a MolGraph object
        representing a molecule. Bond types are represented as follows:
        0 - No Bond
        1 - Single
        2 - Double
        3 - Triple
        4 - Aromatic
        :return: adjacency matrix of a restricted shape as np ndarray
        """
        graph_size = len(self.x)
        bonds_size = len(self.edge_attr)

        if padded:
            adjacency_matrix = np.zeros(
                (DIMENSION, DIMENSION, NUM_BOND_TYPES), dtype=np.float32
            )
        else:
            adjacency_matrix = np.zeros(
                (graph_size, graph_size, NUM_BOND_TYPES), dtype=np.float32
            )

        adjacency_matrix[:, :, 0] = 1

        if graph_size > DIMENSION:
            raise ValueError(f"The graph should have not more than {DIMENSION} nodes")
        if self.edge_attr is None:
            raise ValueError(f"Bond types should be specified in edge_attr of Data")

        edge_attr = self.edge_attr.astype(np.int64)

        for i in range(bonds_size):
            x = self.edge_index[0][i]
            y = self.edge_index[1][i]

            adjacency_matrix[x][y][0] = 0
            adjacency_matrix[y][x][0] = 0

            adjacency_matrix[x][y][edge_attr[i]] = 1
            adjacency_matrix[y][x][edge_attr[i]] = 1

        return adjacency_matrix

    def to_rdkit_mol(self):
        rw_mol = Chem.RWMol()
        atom_indexes = []

        atoms = self.x.tolist()
        bond_index = self.edge_index.tolist()
        bond_attr = self.edge_attr.tolist()

        for atom in atoms:
            idx = rw_mol.AddAtom(Chem.Atom(elements_dict[atom[0]]))
            atom_indexes.append(idx)

        for i, bond in enumerate(bond_index[0]):
            try:
                rw_mol.AddBond(
                    atom_indexes[bond_index[0][i]],
                    atom_indexes[bond_index[1][i]],
                    bond_type_dict[bond_attr[i]],
                )
            except:
                pass

        mol = rw_mol.GetMol()
        return mol

    def elements_vector(self) -> np.ndarray:
        """
        Returns a fixed-sized elements vector
        :return: [atomic_num, ...0...] size(DIMENSION, 1)
        """
        elements_vector = np.zeros(DIMENSION, dtype=np.int64)

        for i in range(len(self.x)):
            elements_vector[i] = self.x[i]

        return elements_vector

    def one_hot_elements_encoding(self, max_n_nodes) -> np.ndarray:
        """
        Returns a one-hot encoded fixed-sized elements vector;
        the number of types is the length of PERMITTED ELEMENTS set
        :return: [, ...0...] size(DIMENSION, len(PERMITTED_ELEMENTS), 1)
        """
        one_hot = np.zeros((max_n_nodes, len(elements_decoder.keys())), dtype=np.int64)

        for i in range(len(self.x)):
            atom_type = elements_decoder[self.x[i].item()]
            one_hot[i][atom_type] = 1

        return one_hot


def prepare_masks_onnx(
    n_nodes: np.ndarray,
    max_n_nodes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare node and edge masks torch-free
    :param n_nodes: a list of sizes of requested molecules (n_samples, 1)
    :param max_n_nodes: maximal number of nodes
    :return: node_mask, edge_mask
    """

    batch_size = n_nodes.shape[0]

    node_mask = np.zeros((batch_size, max_n_nodes))
    for i in range(batch_size):
        node_mask[i, 0 : n_nodes[i]] = 1

    # Compute edge_mask
    edge_mask = np.expand_dims(node_mask, 1) * np.expand_dims(node_mask, 2)
    diag_mask = ~np.eye(edge_mask.shape[1], dtype=bool)[None, :, :]
    edge_mask *= diag_mask
    edge_mask = edge_mask.reshape(-1, 1)
    node_mask = np.expand_dims(node_mask, 2)

    return node_mask, edge_mask


def prepare_edm_input_onnx(
    n_samples: int,
    reference_context: np.ndarray,
    context_norms: dict,
    min_n_nodes: int,
    max_n_nodes: int,
):
    # Create a random list of sizes between min_n_nodes and max_n_nodes of length n_samples
    nodesxsample = np.random.randint(min_n_nodes, max_n_nodes + 1, size=n_samples)

    batch_size = nodesxsample.shape[0]

    node_mask, edge_mask = prepare_masks_onnx(
        n_nodes=nodesxsample, max_n_nodes=max_n_nodes
    )

    normed_context = (reference_context - context_norms["mean"]) / context_norms["mad"]

    batch_context = np.repeat(normed_context[None, :], batch_size, axis=0)

    batch_context = (
        np.repeat(batch_context[:, None, :], max_n_nodes, axis=1) * node_mask
    )

    return (
        node_mask,
        edge_mask,
        batch_context,
    )


def samples_to_rdkit_mol_onnx(
    positions,
    one_hot: np.ndarray,
    node_mask: np.ndarray = None,
    atom_decoder: dict = None,
) -> List[Chem.Mol]:
    """
    Convert EDM Samples to RDKit mol objects torch-free
    :param positions: coordinates tensor
    :param one_hot: one-hot encoded atom types
    :param node_mask: node mask
    :param atom_decoder: atom decoder dictionary
    :return: a list of samples as RDKit Mol objects without bond information
    """
    rdkit_mols = []

    if node_mask is not None:
        atomsxmol = np.sum(node_mask, axis=1)
    else:
        atomsxmol = [one_hot.shape[1]] * one_hot.shape[0]

    for batch_i in range(one_hot.shape[0]):
        xyz_block = "%d\n\n" % atomsxmol[batch_i]
        atoms = np.argmax(one_hot[batch_i], axis=1)
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


def prepare_adj_mat_seer_input_onnx(
    mols: List[Chem.Mol],
    dimension: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Chem.Mol]]:
    canonicalised_samples = []
    n_samples = len(mols)

    elements_batch = np.zeros((n_samples, dimension), dtype=np.int64)
    dist_mat_batch = np.zeros((n_samples, dimension, dimension), dtype=np.float32)
    adj_mat_batch = np.zeros((n_samples, dimension, dimension), dtype=np.float32)

    for i, sample in enumerate(mols):
        mol = canonicalise(sample)

        conf = mol.GetConformer()
        coord = np.array(conf.GetPositions(), dtype=np.float32)

        structure = MolGraphONNX.from_mol(mol=mol, remove_hs=False)
        elements = structure.elements_vector()
        n_atoms = np.count_nonzero(elements)

        target_adjacency_matrix = structure.adjacency_matrix()

        sc_adj_mat = np.argmax(target_adjacency_matrix, axis=2) + np.eye(dimension)

        sc_adj_mat[sc_adj_mat > 0] = 1

        dist_mat = distance_matrix(coord)
        pad_width = ((0, dimension - n_atoms), (0, dimension - n_atoms))
        pad_dist_mat_sc = np.pad(
            dist_mat, pad_width, mode="constant", constant_values=0
        )
        pad_dist_mat_sc += np.eye(dimension)

        elements_batch[i] = elements.astype(np.int64)
        dist_mat_batch[i] = pad_dist_mat_sc
        adj_mat_batch[i] = sc_adj_mat
        canonicalised_samples.append(mol)

    return elements_batch, dist_mat_batch, adj_mat_batch, canonicalised_samples


def prepare_fragment_onnx(
    n_samples: int,
    fragment: Chem.Mol,
    max_n_nodes: int = DIMENSION,
    min_n_nodes: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Torch-free preparation of a Fixed Fragment for Inpainting. Converts Mol to latent Z tensor, ready for injection
    :param n_samples: required batch size of the prepared latent fragment - number of molecules to generate
    :param fragment: fragment to prepare rdkit Mol
    :param max_n_nodes: possible maximum number of nodes - for padding - int
    :param min_n_nodes: possible minimum number of nodes - int
    :return: Latent representation of the fragment and a mask,
             indicating which atoms in the latent representation are fixed
    """

    # Remove Hs
    coord, h = ifm_get_xh_from_fragment_onnx(fragment)
    n_atoms = coord.shape[0]

    # Check that fragment size is adequate
    if n_atoms >= min_n_nodes:
        raise ValueError(
            "Fragment must contain fewer atoms than minimum generation size."
        )
    if n_atoms >= max_n_nodes:
        raise ValueError(
            "Fragment has more atoms than the maximum number of atoms requested."
        )

    x = np.pad(coord, ((0, max_n_nodes - n_atoms), (0, 0)), mode="constant")
    h = np.pad(h, ((0, max_n_nodes - n_atoms), (0, 0)), mode="constant")

    # Batch x and h

    x = np.tile(x[None, :, :], (n_samples, 1, 1))  # (n_samples, max_n_nodes, 3)
    h = np.tile(h[None, :, :], (n_samples, 1, 1))
    z_known = np.concatenate([x, h], axis=2)

    fixed_mask = np.zeros((n_samples, max_n_nodes, 1), dtype=np.float32)
    fixed_mask[:, :n_atoms, 0] = 1.0

    return z_known, fixed_mask


def redefine_bonds_onnx(mol: Chem.Mol, adj_mat: np.ndarray) -> Chem.Mol:
    n = mol.GetNumAtoms()
    # Pass the molecule through xyz block to remove bonds and all extra atom properties
    i_xyz = Chem.MolToXYZBlock(mol)
    c_mol = Chem.MolFromXYZBlock(i_xyz)
    ed_mol = Chem.EditableMol(c_mol)

    repr_m = np.tril(np.argmax(adj_mat, axis=2))
    repr_m = repr_m * (1 - np.eye(repr_m.shape[0], dtype=int))

    for i in range(n):
        for j in range(n):
            # Find out the bond type by indexing 1 in the matrix bond
            bond_type = repr_m[i, j].item()

            if bond_type != 0:
                ed_mol.AddBond(i, j, bond_type_dict[bond_type])

    new_mol = ed_mol.GetMol()

    return new_mol


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


def distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """
    Generates a distance matrices from a xyz coordinates tensor
    :param coordinates: xyz coordinates tensor
    :return: distance matrix
    """
    n = coordinates.shape[0]
    i_mat = np.repeat(coordinates[np.newaxis, :, :], n, axis=0)
    # Repeat coordinates tensor along new dimension
    j_mat = np.transpose(i_mat, (1, 0, 2))

    dist_matrix = np.sqrt(np.sum((i_mat - j_mat) ** 2, axis=2))

    return dist_matrix


def get_context_shape_onnx(coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the principal axes for the conformer,
    and calculates Moment of Inertia tensor for the conformer in principal axes.
    All atom masses are considered equal to one, to capture shape only.
    :param coord: initial coordinates of the atoms
    :return: Principal components of MOI tensor, and coordinates rotated to a principal frame as a tuple of ndarrays
    """
    masses = np.ones(coord.shape[0])
    moi_tensor = get_moment_of_inertia_tensor_onnx(coord, masses)
    # Diagonalize the MOI tensor using eigen decomposition
    _, eigenvectors = np.linalg.eigh(moi_tensor)

    # Rotate points to principal axes
    rotated_points = np.matmul(coord.astype(np.float32), eigenvectors)

    # Get the three main moments of inertia from the main diagonal
    context = np.diag(get_moment_of_inertia_tensor_onnx(rotated_points, masses))

    return context, rotated_points


def get_moment_of_inertia_tensor_onnx(
    coord: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """
    Calculate a Moment of Inertia tensor
    :return: Moment of Inertia Tensor in input coordinates
    """
    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]

    # Diagonal elements
    i_xx = np.sum(weights * (y**2 + z**2))
    i_yy = np.sum(weights * (x**2 + z**2))
    i_zz = np.sum(weights * (x**2 + y**2))

    # Off-diagonal elements
    i_xy = -np.sum(x * y)
    i_xz = -np.sum(x * z)
    i_yz = -np.sum(y * z)

    # Construct the MOI tensor
    moi_tensor = np.array(
        [[i_xx, i_xy, i_xz], [i_xy, i_yy, i_yz], [i_xz, i_yz, i_zz]],
        dtype=np.float32,
    )

    return moi_tensor


def ifm_get_xh_from_fragment_onnx(
    fixed_fragment: Chem.Mol
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get coordinates and atom types as tensors for a fragment torch-free
    :param fixed_fragment: fragment as rdkit Mol object
    :return: fixed_fragment_x, fixed_fragment_h
    """
    # Get coordinates of the fixed fragment
    ff_mol = Chem.RemoveAllHs(fixed_fragment)
    ff_conformer = ff_mol.GetConformer()
    ff_x = np.array(ff_conformer.GetPositions(), dtype=np.float32)

    # Get atom types of a fixed fragment
    ff_structure = MolGraphONNX.from_mol(mol=ff_mol, remove_hs=True)
    ff_n_atoms = ff_x.shape[0]

    ff_h = ff_structure.one_hot_elements_encoding(
        ff_n_atoms
    )  # Atom types of a fixed fragment

    return ff_x, ff_h


def ifm_prepare_gen_fragment_context_onnx(
    fixed_fragment_x: np.ndarray,
    reference_context: np.ndarray,
    context_norms: dict,
    n_nodes: np.ndarray,
    max_n_nodes: int,
    min_n_nodes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare Contexts for Generation of Individual Fragments based on Reference and Fixed Fragment MOI tensors.
    Torch-free.
    Fixed Fragment should be expressed in the same coordinate system as the reference, The origin of such coordinate
    system should be place in the COM of the Reference.

    :param fixed_fragment_x:  coordinates of the fixed fragment atoms
    :param reference_context: Principal components of MOI Tensor of the reference shape (3)
    :param context_norms: Values to norm the context for individual fragment generation
    :param n_nodes: a list of sizes of requested molecules (n_samples, 1)
    :param max_n_nodes: a maximal number of nodes allowed for the whole generated molecule
    :param min_n_nodes: a minimal number of nodes allowed for the whole generated molecule
    :return: frag_node_mask, frag_edge_mask, batched_normed_frag_context, shift, rotation
    """
    batch_size = n_nodes.shape[0]

    ff_n_atoms = fixed_fragment_x.shape[0]
    masses_ff = np.ones(ff_n_atoms)

    if ff_n_atoms >= min_n_nodes:
        raise ValueError(
            "Fragment must contain fewer atoms than minimum generation size."
        )
    if ff_n_atoms >= max_n_nodes:
        raise ValueError(
            "Fragment has more atoms than the maximum number of atoms requested."
        )

    # Fixed fragment MOI around origin
    moi_ff = get_moment_of_inertia_tensor_onnx(fixed_fragment_x, masses_ff)  # (3, 3)

    # Reference MOI as diagonal matrix (expand to B)
    moi_ref = np.diag(reference_context)  # (3, 3)

    moi_ref_batch = np.tile(moi_ref[None, :, :], (batch_size, 1, 1))  # (B, 3, 3)
    moi_ff_batch = np.tile(moi_ff[None, :, :], (batch_size, 1, 1))  # (B, 3, 3)

    # Gen frag MOI around origin
    moi_gen_origin = moi_ref_batch - moi_ff_batch  # (B, 3, 3)

    # COM of fixed fragment
    com_ff = fixed_fragment_x.mean(axis=0)  # (3,)

    gen_n_atoms = n_nodes.reshape(batch_size, 1).astype(float) - ff_n_atoms  # (B, 1)

    # COM of generated fragments in respect to number of atoms in each
    shift = (ff_n_atoms * com_ff.reshape(1, 3)) / gen_n_atoms  # (B, 3)

    # Shift MOI to COM of generated fragment
    moi_gen_com = shift_moi_to_com_batch_onnx(
        moi_gen_origin, shift, gen_n_atoms
    )  # (B, 3, 3)

    # Diagonalize MOI of generated fragment
    frag_context, rotation = np.linalg.eigh(moi_gen_com)  # (B, 3), (B, 3, 3)
    normed_frag_context = (
        (frag_context - context_norms["mean"]) / context_norms["mad"]
    )

    max_n_nodes_frag = max_n_nodes - ff_n_atoms

    # Flatten gen_n_atoms and convert to int
    gen_n_atoms = gen_n_atoms.reshape(-1).astype(np.int64)

    frag_node_mask, frag_edge_mask = prepare_masks_onnx(
        n_nodes=gen_n_atoms,
        max_n_nodes=max_n_nodes_frag,
    )

    batched_normed_frag_context = (
            np.repeat(normed_frag_context[:, None, :], max_n_nodes_frag, axis=1) * frag_node_mask
    )

    return frag_node_mask, frag_edge_mask, batched_normed_frag_context, shift, rotation


def ifm_prepare_fragments_for_merge_onnx(
    fixed_fragment_x: np.ndarray,
    fixed_fragment_h: np.ndarray,
    gen_fragments_x: np.ndarray,
    gen_fragments_h: np.ndarray,
    max_n_nodes: int,
):
    """
    Prepares Multiple Fragments for Merge. Prepares latent Z tensor, ready for injection and mask for a fixed fragment.
    Torch-free
    :param fixed_fragment_x: Coordinates of a fixed fragment as np.ndarray
    :param fixed_fragment_h: One-hot encoded atom types of a fixed fragment as np.ndarray
    :param gen_fragments_x: Batch of coordinates of generated fragments - np.ndarray
    :param gen_fragments_h: Batch of One-hot encoded atom types of generated fragments - np.ndarray
    :param max_n_nodes: maximal allowable number of atoms
    :return: z_known, fixed_mask, n_samples
    """

    n_samples = gen_fragments_x.shape[0]

    ff_n_atoms = fixed_fragment_x.shape[0]

    # Add a batch dimension
    ff_x = fixed_fragment_x[None, :, :]  # Shape: (1, N, 3)
    ff_h = fixed_fragment_h[None, :, :]  # Shape: (1, N, F)

    # Repeat across batch dimension
    ff_x_batched = np.repeat(ff_x, n_samples, axis=0)  # Shape: (n_samples, N, 3)
    ff_h_batched = np.repeat(ff_h, n_samples, axis=0)

    x_prep = np.concatenate([ff_x_batched, gen_fragments_x], axis=1)
    h_prep = np.concatenate([ff_h_batched, gen_fragments_h], axis=1)

    z_known = np.concatenate([x_prep, h_prep], axis=2)

    # The fixed fragment is always in the first place - so we set fixed mask to have 1s only
    # on the first ff_n_atoms elements of z_known

    fixed_mask = np.zeros((n_samples, max_n_nodes, 1), dtype=np.float32)
    fixed_mask[:, :ff_n_atoms, 0] = 1.0

    return z_known, fixed_mask


def shift_moi_to_com_batch_onnx(
    moi_origin: np.ndarray, r_coms: np.ndarray, masses: np.ndarray
) -> np.ndarray:
    """
    Translates moment of inertia from the origin to multiple guessed centers of mass
    using the inverse parallel axis theorem. Torch-Free

    :param moi_origin: (3, 3) Inertia tensor around origin (shared across batch)
    :param r_coms: (B, 3) Vectors from origin to guessed COMs
    :param masses: (B,) Total masses per example
    :return: I_coms: (B, 3, 3) Inertia tensors about the guessed COMs
    """
    batch_size = r_coms.shape[0]
    i_3 = np.tile(np.eye(3)[None, :, :], (batch_size, 1, 1))  # (B, 3, 3)

    r = r_coms.reshape(batch_size, 3, 1)  # (B, 3, 1)
    r_outer = np.matmul(r, np.transpose(r, (0, 2, 1)))  # (B, 3, 3)
    r_norm_sq = np.sum(r_coms ** 2, axis=1).reshape(batch_size, 1, 1)  # (B, 1, 1)

    masses = masses.reshape(batch_size, 1, 1)  # (B, 1, 1)
    shift = masses * (r_norm_sq * i_3 - r_outer)  # (B, 3, 3)

    return moi_origin - shift   # (B, 3, 3)


def inverse_coord_transform_onnx(
    coord: np.ndarray, shift: np.ndarray, rotation: np.ndarray
) -> np.ndarray:
    """
    Inverse shift and Rotation transformation to a batch of xyz coordinates sets. Torch-free
    :param coord: Batch of Coordinates to be modified (batch_size, N, 3)
    :param shift: Batch of shifts to be applied (batch_size, 3)
    :param rotation: Batch of Rotations to be applied (batch_size, 3, 3)
    :return: Modified Coordinates
    """
    # Rotate first
    batch_size = coord.shape[0]
    x_rotated = np.matmul(coord, np.transpose(rotation, (0, 2, 1)))
    # Translate second
    x_translated = x_rotated - shift.reshape(batch_size, 1, 3)

    return x_translated
