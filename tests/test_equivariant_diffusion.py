import torch

from mlconfgen.egnn import EGNNDynamics
from mlconfgen.equivariant_diffusion import (
    EquivariantDiffusion,
    PredefinedNoiseSchedule,
    align_fragment_com_to_generated,
    clip_noise_schedule,
    polynomial_schedule,
)


# --- Free functions ---


def test_clip_noise_schedule_bounds():
    alphas2 = torch.tensor([0.9, 0.8, 0.5, 0.1])
    clip_value = 0.001
    result = clip_noise_schedule(alphas2, clip_value=clip_value)
    assert (result >= clip_value - 1e-7).all()
    assert (result <= 1.0 + 1e-7).all()


def test_polynomial_schedule_length():
    timesteps = 50
    result = polynomial_schedule(timesteps)
    assert len(result) == timesteps + 1


def test_polynomial_schedule_bounds():
    result = polynomial_schedule(100)
    assert (result > 0).all()
    assert (result <= 1.0 + 1e-7).all()


def test_polynomial_schedule_decreasing():
    result = polynomial_schedule(100)
    diffs = result[1:] - result[:-1]
    assert (diffs <= 1e-6).all()


def test_align_fragment_com():
    torch.manual_seed(42)
    B, N = 2, 6
    z_known = torch.randn(B, N, 11)
    z_gen = torch.randn(B, N, 11)
    fixed_mask = torch.zeros(B, N, 1)
    fixed_mask[:, :3, :] = 1.0

    result = align_fragment_com_to_generated(z_known, z_gen, fixed_mask)

    # COM of fixed region in result should match COM of fixed region in z_gen
    coords_result = result[:, :, :3]
    coords_gen = z_gen[:, :, :3]

    com_result = (coords_result * fixed_mask).sum(dim=1) / fixed_mask.sum(dim=1)
    com_gen = (coords_gen * fixed_mask).sum(dim=1) / fixed_mask.sum(dim=1)
    assert torch.allclose(com_result, com_gen, atol=1e-4)


# --- Model shapes ---


def test_noise_schedule_monotonic():
    schedule = PredefinedNoiseSchedule(timesteps=10, precision=1e-4)
    gamma = schedule.gamma.data
    diffs = gamma[1:] - gamma[:-1]
    assert (diffs >= -1e-5).all()


def test_noise_schedule_forward_shape():
    schedule = PredefinedNoiseSchedule(timesteps=10, precision=1e-4)
    t = torch.tensor([0.0, 0.5, 1.0])
    out = schedule(t)
    assert out.shape == t.shape


def test_diffusion_init_no_crash():
    torch.manual_seed(42)
    n_atom_types = 8
    context_node_nf = 3
    dynamics = EGNNDynamics(
        in_node_nf=n_atom_types + 1,
        context_node_nf=context_node_nf,
        hidden_nf=32,
    )
    edm = EquivariantDiffusion(
        dynamics=dynamics,
        in_node_nf=n_atom_types,
        timesteps=10,
    )
    assert edm is not None


def test_diffusion_phi_shape():
    torch.manual_seed(42)
    B, N = 1, 5
    n_atom_types = 8
    context_node_nf = 3
    # EGNNDynamics in_node_nf includes +1 for time
    dynamics = EGNNDynamics(
        in_node_nf=n_atom_types + 1,
        context_node_nf=context_node_nf,
        hidden_nf=32,
    )
    edm = EquivariantDiffusion(
        dynamics=dynamics,
        in_node_nf=n_atom_types,
        timesteps=10,
    )
    # xh has 3 coords + n_atom_types features
    x = torch.randn(B, N, 3 + n_atom_types)
    t = torch.tensor([0.5])
    node_mask = torch.ones(B, N, 1)
    edge_mask = torch.ones(B * N * N, 1)
    context = torch.randn(B, N, context_node_nf)
    out = edm.phi(x, t, node_mask, edge_mask, context)
    assert out.shape == (B, N, 3 + n_atom_types)


def test_diffusion_snr_positive():
    gamma = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    snr = EquivariantDiffusion.snr(gamma)
    assert (snr > 0).all()
