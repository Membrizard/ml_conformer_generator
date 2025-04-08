from typing import Tuple

import numpy
import torch
from rdkit import Chem
from rdkit.Chem import rdmolops

from .config import DIMENSION, NUM_BOND_TYPES

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
bonds_dict = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}


class MolGraph:
    """
    A class to handle molecular graphs using torch.Tensors:
    """

    def __init__(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr

    @classmethod
    def from_adjacency_matrix(
        cls,
        nodes: torch.Tensor,
        adjacency_matrix: torch.Tensor,
    ) -> "MolGraph":
        """
        Create a Molecular Graph from Nodes tensor and given 0-1 normalised adjacency matrix of a proper dimension.
        :param nodes: [[atomic_num, shielding]..] dtype = torch.float
        :param adjacency_matrix: torch tensor of size (DIMENSION, DIMENSION)
        :return: MolGraph object
        """
        if nodes is None:
            raise ValueError(f"Either Nodes tensor or Atom Matrix should be specified.")

        n = len(nodes)

        if adjacency_matrix is None:
            raise ValueError(f"Adjacency matrix should be Specified")

        if adjacency_matrix.size() != torch.Size(
            [DIMENSION, DIMENSION, NUM_BOND_TYPES]
        ):
            raise ValueError(
                f"Adjacency matrix should be of size {DIMENSION} with bond encoding with size of {NUM_BOND_TYPES}"
            )

        edge_index = [[], []]
        edge_attr = []

        repr_m = torch.argmax(adjacency_matrix, dim=2)

        for i in range(n):
            for j in range(n):
                # Find out the bond type by indexing 1 in the matrix bond
                bond_type = repr_m[i, j]

                if bond_type != 0:
                    edge_index[0].append(i)
                    edge_index[1].append(j)
                    edge_attr.append(bond_type)

        return cls(
            x=nodes,
            edge_index=torch.tensor(edge_index),
            edge_attr=torch.tensor(edge_attr),
        )

    @classmethod
    def from_mol(cls, mol: Chem.Mol, remove_hs: bool = True) -> "MolGraph":
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

        x = torch.tensor(out, dtype=torch.float)

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
            edge_index = torch.tensor(numpy.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = torch.tensor(
                numpy.array(edge_features_list, dtype=float), dtype=torch.float
            )
        else:  # mol has no bonds
            raise ValueError(
                f"Bonds must be specified for the molecule - {mol.GetProp('_Name')}."
            )

        return cls(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def adjacency_matrix(self, padded: bool = True) -> torch.tensor:
        """
        Creates a 0-1 normalised adjacency matrix with a specified size from a MolGraph object
        representing a molecule. Bond types are represented as follows:
        0 - No Bond
        1 - Single
        2 - Double
        3 - Triple
        4 - Aromatic
        :return: adjacency matrix of a restricted shape as torch tensor
        """
        graph_size = len(self.x)
        bonds_size = len(self.edge_attr)

        if padded:
            adjacency_matrix = torch.zeros(
                DIMENSION, DIMENSION, NUM_BOND_TYPES, dtype=torch.float
            )
        else:
            adjacency_matrix = torch.zeros(
                graph_size, graph_size, NUM_BOND_TYPES, dtype=torch.float
            )

        adjacency_matrix[:, :, 0] = 1

        if graph_size > DIMENSION:
            raise ValueError(f"The graph should have not more than {DIMENSION} nodes")
        if self.edge_attr is None:
            raise ValueError(f"Bond types should be specified in edge_attr of Data")

        for i in range(bonds_size):
            x = self.edge_index[0][i]
            y = self.edge_index[1][i]

            adjacency_matrix[x][y][0] = 0
            adjacency_matrix[y][x][0] = 0

            adjacency_matrix[x][y][self.edge_attr[i].long().item()] = 1
            adjacency_matrix[y][x][self.edge_attr[i].long().item()] = 1

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
                    bonds_dict[bond_attr[i]],
                )
            except:
                pass

        mol = rw_mol.GetMol()
        return mol

    def elements_vector(self) -> torch.Tensor:
        """
        Returns a fixed-sized elements vector
        :return: [atomic_num, ...0...] size(DIMENSION, 1)
        """
        elements_vector = torch.zeros(DIMENSION, dtype=torch.long)

        for i in range(len(self.x)):
            elements_vector[i] = self.x[i]

        return elements_vector


def vector_graph_sort(
    elements: torch.Tensor,
    coordinates: torch.Tensor,
    adjacency_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A function to efficiently sort the graph nodes order and
    rebuilds the adjacency matrix according to the sorted order
    :param elements: torch tensor (BATCH_SIZE, DIMENSION) -> elements vector
    :param coordinates: torch tensor (BATCH_SIZE, DIMENSION, 3) -> coordinates
    :param adjacency_matrix: (BATCH_SIZE, DIMENSION, DIMENSION, NUM_BOND_TYPES) ->
                                      adjacency matrix corresponding to order
    :return: sorted atoms_matrix and adjacency matrix as a tuple
    """
    # Sort the molecular graphs after generating shielding on GPU
    size = adjacency_matrix.size()
    batch_size = size[0]
    N = size[1]
    num_classes = size[-1]

    # Calculate distances to each atom from the center of mass (oroging)
    distances = torch.sum(torch.pow(coordinates, 2), dim=-1)

    nodes_batch = torch.stack((elements, distances), 2)
    sorted_indexes = torch.sort(torch.sum(nodes_batch, dim=2), descending=True)[1]

    sorted_elements = elements[torch.arange(batch_size).unsqueeze(1), sorted_indexes]
    sorted_coordinates = coordinates[
        torch.arange(batch_size).unsqueeze(1), sorted_indexes
    ]

    # Rearrange the adjacency matrix
    flattened_adjacency_matrix = torch.argmax(adjacency_matrix, dim=3)
    findex = sorted_indexes.repeat_interleave(N, 1) * N + sorted_indexes.repeat(1, N)
    sorted_matrix = (
        flattened_adjacency_matrix.flatten(1, 2)
        .gather(1, findex)
        .view(batch_size, N, N)
    )

    target = torch.nn.functional.one_hot(sorted_matrix, num_classes).type(torch.float)

    return sorted_elements, sorted_coordinates, target
