import torch

from src.mlconfgen.cheminformatics.shape_similarity import (
    AMPLITUDE, _0th_moment_integral, best_pi_rotation_by_tanimoto,
    build_neighbor_sets, find_r_cliques_fast, get_alpha,
    get_shape_quadrupole_for_molecule, i_1st_moment_integral,
    ii_2nd_moment_integral, product_of_n_gaussians, rotate_coord,
    tanimoto_score)

# --- Gaussian math ---


def test_get_alpha_positive():
    alpha = get_alpha()
    assert alpha > 0


def test_0th_moment_positive():
    alpha = get_alpha()
    result = _0th_moment_integral(alpha, AMPLITUDE)
    assert result > 0


def test_0th_moment_scales_with_amplitude():
    alpha = get_alpha()
    v1 = _0th_moment_integral(alpha, 1.0)
    v2 = _0th_moment_integral(alpha, 2.0)
    assert abs(v2 / v1 - 2.0) < 1e-5


def test_1st_moment_zero_at_origin():
    alpha = get_alpha()
    coord = torch.zeros(3, 3)
    result = i_1st_moment_integral(coord, alpha, AMPLITUDE)
    assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)


def test_2nd_moment_positive():
    alpha = get_alpha()
    coord = torch.randn(5, 3)
    result = ii_2nd_moment_integral(coord, alpha, AMPLITUDE)
    assert (result > 0).all()


def test_product_of_gaussians_center_is_mean():
    centers = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    alpha = get_alpha()
    new_center, _, _ = product_of_n_gaussians(centers, alpha, AMPLITUDE)
    expected = torch.tensor([[1.0, 0.0, 0.0]])
    assert torch.allclose(new_center, expected, atol=1e-5)


def test_product_of_gaussians_alpha_scales():
    centers = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    alpha = get_alpha()
    _, new_alpha, _ = product_of_n_gaussians(centers, alpha, AMPLITUDE)
    assert abs(new_alpha - 3 * alpha) < 1e-5


# --- Rotation ---


def test_rotate_coord_identity():
    coord = torch.randn(5, 3)
    angles = torch.zeros(3)
    result = rotate_coord(coord, angles)
    assert torch.allclose(result, coord, atol=1e-5)


def test_rotate_coord_preserves_distances(simple_coords):
    from mlconfgen.utils.mol_utils import distance_matrix

    angles = torch.tensor([0.5, 1.0, 0.3])
    rotated = rotate_coord(simple_coords, angles)
    dm_orig = distance_matrix(simple_coords)
    dm_rot = distance_matrix(rotated)
    assert torch.allclose(dm_orig, dm_rot, atol=1e-4)


# --- Tanimoto ---


def test_tanimoto_self_similarity(paba_coords):
    score = tanimoto_score(paba_coords, paba_coords, n=20)
    assert abs(score - 1.0) < 0.05


def test_tanimoto_range(paba_coords):
    torch.manual_seed(42)
    other = paba_coords + torch.randn_like(paba_coords) * 0.5
    score = tanimoto_score(paba_coords, other, n=20)
    assert 0.0 <= score <= 1.0 + 0.01


def test_tanimoto_symmetry(paba_coords):
    torch.manual_seed(42)
    other = paba_coords + torch.randn_like(paba_coords) * 0.5
    s1 = tanimoto_score(paba_coords, other, n=20)
    s2 = tanimoto_score(other, paba_coords, n=20)
    assert abs(s1 - s2) < 0.05


# --- Cliques ---


def test_find_cliques_complete_graph():
    # K4: fully connected 4-node graph
    adj = torch.ones(4, 4)
    cliques = find_r_cliques_fast(adj, clique_order=2)
    # C(4,2) = 6
    assert cliques.shape[0] == 6
    assert cliques.shape[1] == 2


def test_find_cliques_no_edges():
    adj = torch.eye(4)
    cliques = find_r_cliques_fast(adj, clique_order=2)
    assert cliques.shape[0] == 0


def test_build_neighbor_sets_length():
    adj = torch.ones(5, 5)
    masks = build_neighbor_sets(adj)
    assert len(masks) == 5


# --- Shape quadrupole ---


def test_quadrupole_output_shapes(paba_coords):
    moments, points = get_shape_quadrupole_for_molecule(paba_coords)
    assert moments.shape == (3,)
    assert points.shape == paba_coords.shape


def test_quadrupole_moments_sorted(paba_coords):
    moments, _ = get_shape_quadrupole_for_molecule(paba_coords)
    assert moments[0] >= moments[1] - 1e-5
    assert moments[1] >= moments[2] - 1e-5


# --- best_pi_rotation_by_tanimoto ---


def test_best_pi_rotation_returns_tuple(paba_coords):
    result = best_pi_rotation_by_tanimoto(paba_coords, paba_coords)
    assert isinstance(result, tuple)
    assert len(result) == 2
    best_coord, best_score = result
    assert best_coord.shape == paba_coords.shape
    assert isinstance(best_score, float)


def test_best_pi_rotation_self_similarity(paba_coords):
    _, score = best_pi_rotation_by_tanimoto(paba_coords, paba_coords)
    assert score > 0.9


def test_best_pi_rotation_improves_or_keeps_score(paba_coords):
    # Rotate candidate by pi around y-axis so it needs correction
    pi_y = torch.tensor([0.0, torch.pi, 0.0])
    rotated = rotate_coord(paba_coords, pi_y)
    baseline = tanimoto_score(paba_coords, rotated)
    _, best_score = best_pi_rotation_by_tanimoto(paba_coords, rotated)
    assert best_score >= baseline - 1e-6


def test_best_pi_rotation_custom_tanimoto_fn(paba_coords):
    calls = []

    def fake_tanimoto(ref, cand):
        calls.append(1)
        return 0.5

    best_coord, score = best_pi_rotation_by_tanimoto(
        paba_coords, paba_coords, tanimoto_fn=fake_tanimoto
    )
    # identity + 3 pi-rotations = 4 calls
    assert len(calls) == 4
    assert score == 0.5
