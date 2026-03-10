import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops

from ..utils.common import (allowable_features, bond_type_dict,
                            elements_decoder, elements_dict)
from ..utils.config import DIMENSION, NUM_BOND_TYPES


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
