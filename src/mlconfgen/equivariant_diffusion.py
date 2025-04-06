from typing import Tuple

import torch
import torch.nn.functional as F

from .egnn import EGNNDynamics


def clip_noise_schedule(
    alphas2: torch.Tensor, clip_value: float = 0.001
) -> torch.Tensor:
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """

    alphas2 = torch.cat((torch.ones(1), alphas2), dim=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = torch.clip(alphas_step, min=clip_value, max=1.0)
    alphas2 = torch.cumprod(alphas_step, dim=0)

    return alphas2


def polynomial_schedule(timesteps: int, s: float = 1e-4, power: int = 2):
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


def remove_mean_with_mask(x, node_mask):
    n = torch.sum(node_mask, 1, keepdim=True)

    mean = torch.sum(x, dim=1, keepdim=True) / n
    x = x - mean * node_mask
    return x


def sample_center_gravity_zero_gaussian_with_mask(
    size: Tuple[int, int, int], device: torch.device, node_mask
):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def sample_gaussian_with_mask(
    size: Tuple[int, int, int], device: torch.device, node_mask
):
    x = torch.randn(size, device=device)

    x_masked = x * node_mask
    return x_masked


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, timesteps: int, precision: float, power: int = 2):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        # Default Schedule - polynomial with power 2

        alphas2 = polynomial_schedule(timesteps, s=precision, power=power)

        sigmas2 = 1 - alphas2

        log_alphas2 = torch.log(alphas2)
        log_sigmas2 = torch.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        self.gamma = torch.nn.Parameter(
            (-log_alphas2_to_sigmas2).float(), requires_grad=False
        )

    def forward(self, t: torch.Tensor):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]


class EquivariantDiffusion(torch.nn.Module):
    """
    The E(n) Diffusion Module.
    """

    def __init__(
        self,
        dynamics: EGNNDynamics,
        in_node_nf: int = 8,
        n_dims: int = 3,
        timesteps: int = 1000,
        noise_precision: float = 1e-4,
        norm_values: Tuple[float, float] = (
            1.0,
            9.0,
        ),  # (1, max number of atom classes)
    ):
        super().__init__()

        self.gamma = PredefinedNoiseSchedule(
            timesteps=timesteps, precision=noise_precision
        )

        # The network that will predict the denoising.
        self.dynamics = dynamics

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims

        self.num_classes = self.in_node_nf

        # Declare time steps-related tensors
        self.T = timesteps
        self.time_steps = torch.flip(
            torch.arange(0, timesteps, device=dynamics.device), dims=[0]
        )

        self.norm_values = norm_values

    def phi(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Denoising pass
        """
        net_out = self.dynamics(t, x, node_mask, edge_mask, context)
        return net_out

    @staticmethod
    def inflate_batch_array(array: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """Computes alpha given gamma."""
        return self.inflate_batch_array(
            torch.sqrt(torch.sigmoid(-gamma)), target_tensor
        )

    @staticmethod
    def snr(gamma: torch.Tensor) -> torch.Tensor:
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    def unnormalize(
        self, x: torch.Tensor, h_cat: torch.Tensor, node_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1]

        h_cat = h_cat * node_mask

        return x, h_cat

    def sigma_and_alpha_t_given_s(
        self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            1 - torch.exp(F.softplus(gamma_s) - F.softplus(gamma_t)), target_tensor
        )

        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def compute_x_pred(
        self, net_out: torch.Tensor, zt: torch.Tensor, gamma_t: torch.Tensor
    ) -> torch.Tensor:
        """Commputes x_pred, i.e. the most likely prediction of x."""

        sigma_t = self.sigma(gamma_t, target_tensor=net_out)
        alpha_t = self.alpha(gamma_t, target_tensor=net_out)
        eps_t = net_out
        x_pred = 1.0 / alpha_t * (zt - sigma_t * eps_t)

        return x_pred

    def sample_p_xh_given_z0(
        self,
        z0: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.snr(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask)

        x = xh[:, :, : self.n_dims]

        x, h_cat = self.unnormalize(x, z0[:, :, self.n_dims : -1], node_mask)

        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h = h_cat
        return x, h

    def sample_normal(
        self, mu: torch.Tensor, sigma: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        """Samples from a Normal distribution."""
        bs = mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps

    def sample_p_zs_given_zt(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        zt: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        (
            sigma2_t_given_s,
            sigma_t_given_s,
            alpha_t_given_s,
        ) = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt)

        sigma_s = self.sigma(gamma_s, target_tensor=zt)
        sigma_t = self.sigma(gamma_t, target_tensor=zt)

        # Neural net prediction.
        eps_t = self.phi(zt, t, node_mask, edge_mask, context)

        mu = (
            zt / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        )

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask)

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [
                remove_mean_with_mask(zs[:, :, : self.n_dims], node_mask),
                zs[:, :, self.n_dims :],
            ],
            dim=2,
        )
        return zs

    def sample_combined_position_feature_noise(
        self, n_samples: int, n_nodes: int, node_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims),
            device=node_mask.device,
            node_mask=node_mask,
        )

        z_h = sample_gaussian_with_mask(
            size=(
                n_samples,
                n_nodes,
                self.in_node_nf,
            ),
            device=node_mask.device,
            node_mask=node_mask,
        )
        z = torch.cat([z_x, z_h], dim=2)
        return z

    def forward(
        self,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw samples from the generative model.
        Inference
        """
        n_samples, n_nodes, _ = node_mask.size()

        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in self.time_steps:
            s_array = torch.full([n_samples, 1], fill_value=s, device=z.device)
            t_array = s_array + 1.0
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s_array,
                t_array,
                z,
                node_mask,
                edge_mask,
                context,
            )

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(
            z,
            node_mask,
            edge_mask,
            context,
        )

        return x, h
