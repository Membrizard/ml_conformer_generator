from typing import Tuple, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor


def clip_noise_schedule(
    alphas2: Tensor, clip_value: float = 0.001
) -> Tensor:
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """

    alphas2 = torch.cat((torch.ones(1), alphas2), dim=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = torch.clip(alphas_step, min=clip_value, max=1.0)
    alphas2 = torch.cumprod(alphas_step, dim=0)

    return alphas2


def polynomial_schedule(
    timesteps: int, s: float = 1e-4, power: int = 2
) -> Tensor:
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.

    Remark - rewritten in torch only
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas2 = (1 - torch.pow(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def remove_mean_with_mask(x: Tensor, node_mask: Tensor) -> Tensor:
    n = torch.sum(node_mask, 1, keepdim=True)

    mean = torch.sum(x, dim=1, keepdim=True) / n
    x = x - mean * node_mask
    return x


def sample_center_gravity_zero_gaussian_with_mask(
    size: Tuple[int, int, int], device: torch.device, node_mask: Tensor,
) -> Tensor:
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def sample_gaussian_with_mask(
    size: Tuple[int, int, int], device: torch.device, node_mask
) -> Tensor:
    x = torch.randn(size, device=device)
    x_masked = x * node_mask
    return x_masked


def align_fragment_com_to_generated(
    z_known_noised: Tensor, z_generated: Tensor, fixed_mask: Tensor
) -> Tensor:
    """
    Aligns COM of the fixed fragment with the corresponding generated fragment during inpainting for equivariance.
    :param z_known_noised: z_known with noise applied
    :param z_generated: z_generated with comparable nois
    :param fixed_mask: a mask to indentify the fixed fragment
    :return: aligned latent representation of a fixed fragment
    """

    coords_known = z_known_noised[:, :, :3]
    coords_gen = z_generated[:, :, :3]

    frag_com_gen = torch.sum(coords_gen * fixed_mask, dim=1, keepdim=True) / (
        fixed_mask.sum(dim=1, keepdim=True)
    )
    frag_com_known = torch.sum(coords_known * fixed_mask, dim=1, keepdim=True) / (
        fixed_mask.sum(dim=1, keepdim=True)
    )

    shift = frag_com_gen - frag_com_known
    coords_shifted = coords_known + shift * fixed_mask  # only move fixed region

    z_known_shifted = z_known_noised.clone()
    z_known_shifted[:, :, :3] = coords_shifted
    return z_known_shifted

def sample_combined_position_feature_noise(
        n_samples: int,
        n_nodes: int,
        node_mask: Tensor,
        in_node_nf: int,
        n_dims: int,
    ) -> Tensor:
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, n_dims),
            device=node_mask.device,
            node_mask=node_mask,
        )

        z_h = sample_gaussian_with_mask(
            size=(
                n_samples,
                n_nodes,
                in_node_nf,
            ),
            device=node_mask.device,
            node_mask=node_mask,
        )
        z = torch.cat([z_x, z_h], dim=2)
        return z