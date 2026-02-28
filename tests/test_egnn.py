import torch

from mlconfgen.egnn import (
    GCL,
    EGNN,
    EGNNDynamics,
    EquivariantBlock,
    coord2diff,
    remove_mean_with_mask,
    unsorted_segment_sum,
)


# --- coord2diff ---


def test_coord2diff_shapes():
    N = 6
    x = torch.randn(N, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    E = edge_index.size(1)
    radial, coord_diff = coord2diff(x, edge_index)
    assert radial.shape == (E, 1)
    assert coord_diff.shape == (E, 3)


def test_coord2diff_radial_non_negative():
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    radial, _ = coord2diff(x, edge_index)
    assert (radial >= 0).all()


# --- unsorted_segment_sum ---


def test_unsorted_segment_sum_known():
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    segment_ids = torch.tensor([0, 1, 0])
    result = unsorted_segment_sum(data, segment_ids, num_segments=2, normalization_factor=1.0)
    expected = torch.tensor([[6.0, 8.0], [3.0, 4.0]])
    assert torch.allclose(result, expected, atol=1e-5)


def test_unsorted_segment_sum_normalization():
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    segment_ids = torch.tensor([0, 1, 0])
    result = unsorted_segment_sum(data, segment_ids, num_segments=2, normalization_factor=2.0)
    expected = torch.tensor([[6.0, 8.0], [3.0, 4.0]]) / 2.0
    assert torch.allclose(result, expected, atol=1e-5)


# --- remove_mean_with_mask ---


def test_remove_mean_with_mask_zero_mean():
    x = torch.randn(2, 5, 3)
    node_mask = torch.ones(2, 5, 1)
    result = remove_mean_with_mask(x, node_mask)
    mean = result.mean(dim=1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)


def test_remove_mean_preserves_masked_out():
    x = torch.randn(1, 4, 3)
    node_mask = torch.tensor([[[1.0], [1.0], [0.0], [0.0]]])
    x = x * node_mask  # zero out masked positions first (as in real usage)
    result = remove_mean_with_mask(x, node_mask)
    assert torch.allclose(result * (1 - node_mask), torch.zeros(1, 4, 3), atol=1e-5)


# --- Model shapes ---


def test_gcl_forward_shape():
    torch.manual_seed(42)
    in_nf, out_nf, hidden_nf = 8, 8, 32  # out_nf must equal in_nf for residual connection
    N = 10
    gcl = GCL(input_nf=in_nf, output_nf=out_nf, hidden_nf=hidden_nf, edges_in_d=1)
    h = torch.randn(N, in_nf)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    edge_attr = torch.randn(4, 1)
    node_mask = torch.ones(N, 1)
    edge_mask = torch.ones(4, 1)
    h_out, _ = gcl(h, edge_index, edge_attr, node_mask, edge_mask)
    assert h_out.shape == (N, out_nf)


def test_equivariant_block_forward_shapes():
    torch.manual_seed(42)
    hidden_nf = 32
    N = 6
    block = EquivariantBlock(hidden_nf=hidden_nf, edge_feat_nf=2)
    h = torch.randn(N, hidden_nf)
    x = torch.randn(N, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]])
    node_mask = torch.ones(N, 1)
    edge_mask = torch.ones(6, 1)
    edge_attr = torch.randn(6, 1)
    h_out, x_out = block(h, x, edge_index, node_mask, edge_mask, edge_attr)
    assert h_out.shape == (N, hidden_nf)
    assert x_out.shape == (N, 3)


def test_egnn_dynamics_forward_shape():
    torch.manual_seed(42)
    B, N = 1, 5
    n_atom_types = 8
    # EGNNDynamics in_node_nf includes +1 for time step
    in_node_nf = n_atom_types + 1
    context_node_nf = 3
    hidden_nf = 32
    device = torch.device("cpu")

    dynamics = EGNNDynamics(
        in_node_nf=in_node_nf,
        context_node_nf=context_node_nf,
        hidden_nf=hidden_nf,
        device=device,
    )
    t = torch.tensor([0.5])
    # xh has 3 coords + n_atom_types features (time is added internally)
    xh = torch.randn(B, N, 3 + n_atom_types)
    node_mask = torch.ones(B, N, 1)
    edge_mask = torch.ones(B * N * N, 1)
    context = torch.randn(B, N, context_node_nf)
    out = dynamics(t, xh, node_mask, edge_mask, context)
    assert out.shape == (B, N, 3 + n_atom_types)


def test_egnn_dynamics_adj_matrix_shape():
    B, N = 2, 4
    device = torch.device("cpu")
    edges = EGNNDynamics.get_adj_matrix(N, B, device)
    assert edges.shape == (2, B * N * N)
