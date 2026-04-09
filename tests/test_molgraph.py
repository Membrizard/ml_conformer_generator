import pytest
import torch

from src.mlconfgen.utils.config import (DIMENSION, NUM_BOND_TYPES,
                                        PERMITTED_ELEMENTS)
from src.mlconfgen.utils.molgraph import MolGraph, vector_graph_sort

# --- MolGraph.from_mol ---


def test_from_mol_returns_molgraph(paba_mol):
    g = MolGraph.from_mol(paba_mol, remove_hs=True)
    assert isinstance(g, MolGraph)


def test_from_mol_x_length(paba_mol_no_hs):
    g = MolGraph.from_mol(paba_mol_no_hs, remove_hs=False)
    assert len(g.x) == paba_mol_no_hs.GetNumAtoms()


def test_from_mol_valid_elements(paba_mol):
    g = MolGraph.from_mol(paba_mol, remove_hs=True)
    for atomic_num in g.x.tolist():
        assert int(atomic_num) in PERMITTED_ELEMENTS


def test_from_mol_edge_index_shape(paba_mol):
    g = MolGraph.from_mol(paba_mol, remove_hs=True)
    assert g.edge_index.shape[0] == 2
    num_edges = g.edge_index.shape[1]
    assert num_edges > 0


def test_from_mol_edges_bidirectional(paba_mol):
    g = MolGraph.from_mol(paba_mol, remove_hs=True)
    edges = set()
    for k in range(g.edge_index.shape[1]):
        i, j = g.edge_index[0, k].item(), g.edge_index[1, k].item()
        edges.add((i, j))
    for i, j in list(edges):
        assert (j, i) in edges


def test_from_mol_remove_hs(paba_mol):
    g_no_hs = MolGraph.from_mol(paba_mol, remove_hs=True)
    g_with_hs = MolGraph.from_mol(paba_mol, remove_hs=False)
    assert len(g_no_hs.x) == 10
    assert len(g_with_hs.x) == 17


# --- adjacency_matrix ---


def test_adj_mat_padded_shape(paba_mol):
    g = MolGraph.from_mol(paba_mol, remove_hs=True)
    adj = g.adjacency_matrix(padded=True)
    assert adj.shape == (DIMENSION, DIMENSION, NUM_BOND_TYPES)


def test_adj_mat_unpadded_shape(paba_mol_no_hs):
    g = MolGraph.from_mol(paba_mol_no_hs, remove_hs=False)
    n = paba_mol_no_hs.GetNumAtoms()
    adj = g.adjacency_matrix(padded=False)
    assert adj.shape == (n, n, NUM_BOND_TYPES)


def test_adj_mat_symmetry(paba_mol):
    g = MolGraph.from_mol(paba_mol, remove_hs=True)
    adj = g.adjacency_matrix(padded=False)
    for k in range(NUM_BOND_TYPES):
        assert torch.allclose(adj[:, :, k], adj[:, :, k].T)


def test_adj_mat_no_bond_default(paba_mol):
    g = MolGraph.from_mol(paba_mol, remove_hs=True)
    adj = g.adjacency_matrix(padded=True)
    n = len(g.x)
    # Padding region should have channel-0 = 1
    assert (adj[n:, n:, 0] == 1).all()


# --- elements_vector ---


def test_elements_vector_shape(paba_mol):
    g = MolGraph.from_mol(paba_mol, remove_hs=True)
    ev = g.elements_vector()
    assert ev.shape == (DIMENSION,)


def test_elements_vector_nonzero_prefix(paba_mol):
    g = MolGraph.from_mol(paba_mol, remove_hs=True)
    ev = g.elements_vector()
    n = len(g.x)
    assert (ev[:n] != 0).all()
    assert (ev[n:] == 0).all()


# --- one_hot_elements_encoding ---


def test_one_hot_shape(paba_mol):
    g = MolGraph.from_mol(paba_mol, remove_hs=True)
    n = len(g.x)
    oh = g.one_hot_elements_encoding(max_n_nodes=n)
    assert oh.shape == (n, len(PERMITTED_ELEMENTS))


def test_one_hot_row_sums(paba_mol):
    g = MolGraph.from_mol(paba_mol, remove_hs=True)
    n = len(g.x)
    max_n = n + 5
    oh = g.one_hot_elements_encoding(max_n_nodes=max_n)
    # Real atom rows sum to 1
    assert (oh[:n].sum(dim=1) == 1).all()
    # Padding rows sum to 0
    assert (oh[n:].sum(dim=1) == 0).all()


# --- from_adjacency_matrix ---


def test_from_adj_matrix_roundtrip(paba_mol):
    g = MolGraph.from_mol(paba_mol, remove_hs=True)
    adj = g.adjacency_matrix(padded=True)
    nodes = g.x.unsqueeze(1)
    g2 = MolGraph.from_adjacency_matrix(nodes=nodes, adjacency_matrix=adj)
    # Same number of edges
    assert g2.edge_index.shape[1] == g.edge_index.shape[1]


def test_from_adj_matrix_wrong_shape_raises():
    nodes = torch.tensor([[6.0], [7.0]])
    bad_adj = torch.zeros(5, 5, NUM_BOND_TYPES)
    with pytest.raises(ValueError):
        MolGraph.from_adjacency_matrix(nodes=nodes, adjacency_matrix=bad_adj)


# --- vector_graph_sort ---


def test_sort_output_shapes():
    torch.manual_seed(42)
    B, N = 2, DIMENSION
    elements = torch.randint(0, 8, (B, N)).float()
    coordinates = torch.randn(B, N, 3)
    adj = torch.zeros(B, N, N, NUM_BOND_TYPES)
    adj[:, :, :, 0] = 1

    se, sc, sa = vector_graph_sort(elements, coordinates, adj)
    assert se.shape == elements.shape
    assert sc.shape == coordinates.shape
    assert sa.shape == adj.shape


def test_sort_preserves_elements():
    torch.manual_seed(42)
    B, N = 1, 10
    elements = torch.tensor([[6, 7, 8, 6, 7, 8, 6, 7, 8, 9]], dtype=torch.float)
    coordinates = torch.randn(B, N, 3)
    adj = torch.zeros(B, N, N, NUM_BOND_TYPES)
    adj[:, :, :, 0] = 1

    se, _, _ = vector_graph_sort(elements, coordinates, adj)
    assert sorted(elements[0].tolist()) == sorted(se[0].tolist())
