import math

import torch

from src.mlconfgen.utils.config import MAX_N_NODES, MIN_N_NODES
from src.mlconfgen.utils.mol_utils import (
    apply_transform, coord_to_pf_batched, distance_matrix, get_context_shape,
    get_moment_of_inertia_tensor, get_moment_of_inertia_tensor_batched,
    inverse_coord_transform, prepare_edm_input, prepare_masks,
    shift_moi_to_com_batch)

# --- distance_matrix ---


def test_distance_matrix_shape(simple_coords):
    dm = distance_matrix(simple_coords)
    n = simple_coords.size(0)
    assert dm.shape == (n, n)


def test_distance_matrix_diagonal_zeros(simple_coords):
    dm = distance_matrix(simple_coords)
    assert torch.allclose(torch.diag(dm), torch.zeros(dm.size(0)), atol=1e-5)


def test_distance_matrix_symmetry(simple_coords):
    dm = distance_matrix(simple_coords)
    assert torch.allclose(dm, dm.T, atol=1e-5)


def test_distance_matrix_known_values(simple_coords):
    dm = distance_matrix(simple_coords)
    assert torch.allclose(dm[0, 1], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(dm[1, 2], torch.tensor(math.sqrt(2)), atol=1e-5)


def test_distance_matrix_non_negative(simple_coords):
    dm = distance_matrix(simple_coords)
    assert (dm >= 0).all()


# --- get_moment_of_inertia_tensor ---


def test_moi_tensor_shape(simple_coords):
    weights = torch.ones(simple_coords.size(0))
    moi = get_moment_of_inertia_tensor(simple_coords, weights)
    assert moi.shape == (3, 3)


def test_moi_tensor_symmetry(simple_coords):
    weights = torch.ones(simple_coords.size(0))
    moi = get_moment_of_inertia_tensor(simple_coords, weights)
    assert torch.allclose(moi, moi.T, atol=1e-5)


def test_moi_tensor_positive_diagonal(simple_coords):
    weights = torch.ones(simple_coords.size(0))
    moi = get_moment_of_inertia_tensor(simple_coords, weights)
    assert (torch.diag(moi) >= 0).all()


def test_moi_tensor_single_atom_at_origin():
    coord = torch.zeros(1, 3)
    weights = torch.ones(1)
    moi = get_moment_of_inertia_tensor(coord, weights)
    assert torch.allclose(moi, torch.zeros(3, 3), atol=1e-5)


# --- get_moment_of_inertia_tensor_batched ---


def test_moi_batched_shape(batch_coords):
    B, N, _ = batch_coords.shape
    weights = torch.ones(N)
    moi = get_moment_of_inertia_tensor_batched(batch_coords, weights)
    assert moi.shape == (B, 3, 3)


def test_moi_batched_matches_unbatched(simple_coords):
    coords = simple_coords.unsqueeze(0)  # (1, 4, 3)
    weights = torch.ones(simple_coords.size(0))
    batched = get_moment_of_inertia_tensor_batched(coords, weights)
    unbatched = get_moment_of_inertia_tensor(simple_coords, weights)
    assert torch.allclose(batched[0], unbatched, atol=1e-5)


# --- get_context_shape ---


def test_context_shape_output_lengths(simple_coords):
    result_no_rot = get_context_shape(simple_coords, include_rotation=False)
    assert len(result_no_rot) == 2

    result_rot = get_context_shape(simple_coords, include_rotation=True)
    assert len(result_rot) == 3


def test_context_shape_principal_components_non_negative(simple_coords):
    context, _ = get_context_shape(simple_coords)
    assert (context >= -1e-5).all()


def test_context_shape_rotation_orthogonal(simple_coords):
    _, _, R = get_context_shape(simple_coords, include_rotation=True)
    eye = torch.eye(3)
    assert torch.allclose(R.T @ R, eye, atol=1e-5)


def test_context_shape_rotated_coords_shape(simple_coords):
    _, rotated = get_context_shape(simple_coords)
    assert rotated.shape == simple_coords.shape


# --- prepare_masks ---


def test_prepare_masks_node_shape(device):
    n_nodes = torch.tensor([3, 5])
    max_n = 6
    node_mask, _ = prepare_masks(n_nodes, max_n, device)
    assert node_mask.shape == (2, max_n, 1)


def test_prepare_masks_edge_shape(device):
    n_nodes = torch.tensor([3, 5])
    max_n = 6
    _, edge_mask = prepare_masks(n_nodes, max_n, device)
    assert edge_mask.shape == (2 * max_n * max_n, 1)


def test_prepare_masks_node_counts(device):
    n_nodes = torch.tensor([3, 5])
    max_n = 6
    node_mask, _ = prepare_masks(n_nodes, max_n, device)
    counts = node_mask.squeeze(-1).sum(dim=1)
    assert torch.allclose(counts, n_nodes.float())


def test_prepare_masks_no_self_loops(device):
    n_nodes = torch.tensor([4])
    max_n = 4
    _, edge_mask = prepare_masks(n_nodes, max_n, device)
    edge_2d = edge_mask.view(max_n, max_n)
    assert (torch.diag(edge_2d) == 0).all()


# --- apply_transform / inverse_coord_transform ---


def test_apply_transform_identity():
    coord = torch.randn(5, 3)
    shift = torch.zeros(3)
    rotation = torch.eye(3)
    result = apply_transform(coord, shift, rotation)
    assert torch.allclose(result, coord, atol=1e-5)


def test_inverse_transform_roundtrip():
    torch.manual_seed(42)
    B, N = 2, 5
    coord = torch.randn(B, N, 3)
    shift = torch.randn(B, 3)
    # Create random orthogonal rotation matrices
    rotation = torch.linalg.qr(torch.randn(B, 3, 3))[0]

    transformed = torch.bmm(coord + shift.unsqueeze(1), rotation)
    recovered = inverse_coord_transform(transformed, shift, rotation)
    assert torch.allclose(recovered, coord, atol=1e-4)


# --- shift_moi_to_com_batch ---


def test_shift_moi_shape():
    moi_origin = torch.eye(3)
    r_coms = torch.randn(3, 3)
    masses = torch.ones(3)
    result = shift_moi_to_com_batch(moi_origin, r_coms, masses)
    assert result.shape == (3, 3, 3)


def test_shift_moi_zero_shift():
    moi_origin = torch.eye(3) * 5.0
    r_coms = torch.zeros(2, 3)
    masses = torch.ones(2)
    result = shift_moi_to_com_batch(moi_origin, r_coms, masses)
    for i in range(2):
        assert torch.allclose(result[i], moi_origin, atol=1e-5)


# --- coord_to_pf_batched ---


def test_pf_batched_shape(batch_coords):
    result = coord_to_pf_batched(batch_coords)
    assert result.shape == batch_coords.shape


def test_pf_batched_centered(batch_coords):
    result = coord_to_pf_batched(batch_coords)
    mean_per_sample = result.mean(dim=1)
    assert torch.allclose(mean_per_sample, torch.zeros_like(mean_per_sample), atol=1e-4)


# --- prepare_edm_input ---


def test_edm_input_shapes(context_norms, device):
    torch.manual_seed(42)
    n_samples = 2
    ref_context = torch.tensor([100.0, 400.0, 500.0])
    node_mask, edge_mask, context = prepare_edm_input(
        n_samples=n_samples,
        reference_context=ref_context,
        context_norms=context_norms,
        min_n_nodes=MIN_N_NODES,
        max_n_nodes=MAX_N_NODES,
        device=device,
    )
    assert node_mask.shape == (n_samples, MAX_N_NODES, 1)
    assert edge_mask.shape == (n_samples * MAX_N_NODES * MAX_N_NODES, 1)
    assert context.shape == (n_samples, MAX_N_NODES, 3)


def test_edm_input_context_masked(context_norms, device):
    torch.manual_seed(42)
    n_samples = 2
    ref_context = torch.tensor([100.0, 400.0, 500.0])
    node_mask, _, context = prepare_edm_input(
        n_samples=n_samples,
        reference_context=ref_context,
        context_norms=context_norms,
        min_n_nodes=MIN_N_NODES,
        max_n_nodes=MAX_N_NODES,
        device=device,
    )
    masked_out = context * (1 - node_mask)
    assert torch.allclose(masked_out, torch.zeros_like(masked_out), atol=1e-5)


def test_edm_input_pad_to_shapes(context_norms, device):
    """When pad_to > max_n_nodes, output tensors use pad_to for the node dimension."""
    torch.manual_seed(42)
    n_samples = 2
    min_n = 10
    max_n = 10
    pad_to = 20
    ref_context = torch.tensor([100.0, 400.0, 500.0])
    node_mask, edge_mask, context = prepare_edm_input(
        n_samples=n_samples,
        reference_context=ref_context,
        context_norms=context_norms,
        min_n_nodes=min_n,
        max_n_nodes=max_n,
        device=device,
        pad_to=pad_to,
    )
    assert node_mask.shape == (n_samples, pad_to, 1)
    assert edge_mask.shape == (n_samples * pad_to * pad_to, 1)
    assert context.shape == (n_samples, pad_to, 3)


def test_edm_input_pad_to_node_counts(context_norms, device):
    """With pad_to, actual node counts should still match min/max_n_nodes, not pad_to."""
    torch.manual_seed(42)
    n_samples = 4
    n_atoms = 8
    pad_to = 15
    ref_context = torch.tensor([100.0, 400.0, 500.0])
    node_mask, _, _ = prepare_edm_input(
        n_samples=n_samples,
        reference_context=ref_context,
        context_norms=context_norms,
        min_n_nodes=n_atoms,
        max_n_nodes=n_atoms,
        device=device,
        pad_to=pad_to,
    )
    counts = node_mask.squeeze(-1).sum(dim=1)
    assert torch.allclose(counts, torch.full((n_samples,), float(n_atoms)))
