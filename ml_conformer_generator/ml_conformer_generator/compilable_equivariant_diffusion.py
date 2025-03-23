import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

from .egnn import EGNNDynamics


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.

    Remark - rewritten in torch only
    """

    alphas2 = torch.cat((torch.ones(1), alphas2), dim=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = torch.clip(alphas_step, min=clip_value, max=1.0)
    alphas2 = torch.cumprod(alphas_step, dim=0)

    return alphas2


def polynomial_schedule(timesteps: int, s=1e-4, power=2):
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


def assert_correctly_masked(variable, node_mask):
    assert (
        variable * (1 - node_mask)
    ).abs().max().item() < 1e-4, "Variables not masked properly."


# def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
#     assert_correctly_masked(x, node_mask)
#     largest_value = x.abs().max().item()
#     error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
#     rel_error = error / (largest_value + eps)
#     assert rel_error < 1e-2, f"Mean is not zero, relative_error {rel_error}"


def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f"Error {masked_max_abs_value} too high"
    # N = node_mask.sum(1, keepdims=True)
    N = torch.sum(node_mask, 1, keepdim=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


def gaussian_KL(q_mu, q_sigma, p_mu, p_sigma, node_mask):
    """Computes the KL distance between two normal distributions.

    Args:
        q_mu: Mean of distribution q.
        q_sigma: Standard deviation of distribution q.
        p_mu: Mean of distribution p.
        p_sigma: Standard deviation of distribution p.
    Returns:
        The KL distance, summed over all dimensions except the batch dim.
    """
    return sum_except_batch(
        (
            torch.log(p_sigma / q_sigma)
            + 0.5 * (q_sigma**2 + (q_mu - p_mu) ** 2) / (p_sigma**2)
            - 0.5
        )
        * node_mask
    )


def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

    Args:
        q_mu: Mean of distribution q.
        q_sigma: Standard deviation of distribution q.
        p_mu: Mean of distribution p.
        p_sigma: Standard deviation of distribution p.
    Returns:
        The KL distance, summed over all dimensions except the batch dim.
    """
    mu_norm2 = sum_except_batch((q_mu - p_mu) ** 2)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return (
        d * torch.log(p_sigma / q_sigma)
        + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2)
        - 0.5 * d
    )


def sample_center_gravity_zero_gaussian_with_mask(size: typing.Tuple[int, int, int], device: torch.device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def sample_gaussian_with_mask(size: typing.Tuple[int, int, int], device: torch.device, node_mask):
    x = torch.randn(size, device=device)

    x_masked = x * node_mask
    return x_masked


# class PositiveLinear(torch.nn.Module):
#     """Linear layer with weights forced to be positive."""
#
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         # bias: bool = True,
#         weight_init_offset: int = -2,
#     ):
#         super(PositiveLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.empty((out_features, in_features)))
#         # if bias:
#         self.bias = nn.Parameter(torch.empty(out_features))
#         # else:
#         #     self.register_parameter("bias", None)
#         self.weight_init_offset = weight_init_offset
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#
#         with torch.no_grad():
#             self.weight.add_(self.weight_init_offset)
#
#         # if self.bias is not None:
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#         torch.nn.init.uniform_(self.bias, -bound, bound)
#
#     def forward(self, input):
#         positive_weight = F.softplus(self.weight)
#         return F.linear(input, positive_weight, self.bias)


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
        in_node_nf: int,
        n_dims: int = 3,
        timesteps: int = 1000,
        noise_precision: float = 1e-4,
        norm_values: typing.Tuple[float, float] = (1.0, 9.0),  # (1, max number of atom classes)
        # norm_biases=(None, 0.0),
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

        # Declare timesteps-related tensors
        self.T = timesteps
        self.timesteps = torch.flip(torch.arange(0, timesteps, device=dynamics.device), dims=[0])

        self.norm_values = norm_values
        # self.norm_biases = norm_biases

        # self.register_buffer("buffer", torch.zeros(1))

        # self.check_issues_norm_values()

    @staticmethod
    def assert_mean_zero_with_mask(x, node_mask, eps: float = 1e-10):
        assert_correctly_masked(x, node_mask)
        largest_value = x.abs().max().item()
        error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        rel_error = error / (largest_value + eps)
        assert rel_error < 1e-2, f"Mean is not zero, relative_error {rel_error}"

    # def check_issues_norm_values(self, num_stdevs: int = 8):
    #     zeros = torch.zeros((1, 1))
    #     gamma_0 = self.gamma(zeros)
    #     sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()
    #
    #     max_norm_value = self.norm_values[1]
    #
    #     if sigma_0 * num_stdevs > 1.0 / max_norm_value:
    #         raise ValueError(
    #             f"Value for normalization value {max_norm_value} probably too "
    #             f"large with sigma_0 {sigma_0:.5f} and "
    #             f"1 / norm_value = {1. / max_norm_value}"
    #         )

    def phi(self, x, t, node_mask, edge_mask, context):
        net_out = self.dynamics(t, x, node_mask, edge_mask, context)

        return net_out

    @staticmethod
    def inflate_batch_array(array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        # print(f"gamma - {gamma}")
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(
            torch.sqrt(torch.sigmoid(-gamma)), target_tensor
        )

    @staticmethod
    def snr(gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)

    # def subspace_dimensionality(self, node_mask):
    #     """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
    #     number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
    #     return (number_of_nodes - 1) * self.n_dims

    # def normalize(self, x, h, node_mask):
    #     x = x / self.norm_values[0]
    #     delta_log_px = -self.subspace_dimensionality(node_mask) * math.log(
    #         self.norm_values[0]
    #     )
    #
    #     # Casting to float in case h still has long or int type.
    #     # h = (h.float() - self.norm_biases[1]) / self.norm_values[1] * node_mask
    #     h = h.float() / self.norm_values[1] * node_mask
    #
    #     return x, h, delta_log_px

    def unnormalize(self, x, h_cat, node_mask):
        x = x * self.norm_values[0]
        # h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * self.norm_values[1]

        h_cat = h_cat * node_mask

        return x, h_cat

    def sigma_and_alpha_t_given_s(
        self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor
    ):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        # sigma2_t_given_s = self.inflate_batch_array(
        #     -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)), target_tensor
        # )
        # Replaced -expm1 with 1 - exp(x) for onnx
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

    # def kl_prior(self, xh, node_mask):
    #     """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).
    #
    #     This is essentially a lot of work for something that is in practice negligible in the loss. However, you
    #     compute it so that you see it when you've made a mistake in your noise schedule.
    #     """
    #     # Compute the last alpha value, alpha_T.
    #     ones = torch.ones((xh.size(0), 1), device=xh.device)
    #     gamma_T = self.gamma(ones)
    #     alpha_T = self.alpha(gamma_T, xh)
    #
    #     # Compute means.
    #     mu_T = alpha_T * xh
    #
    #     mu_T_x, mu_T_h = mu_T[:, :, : self.n_dims], mu_T[:, :, self.n_dims :]
    #
    #     # Compute standard deviations (only batch axis for x-part, inflated for h-part).
    #     sigma_T_x = self.sigma(
    #         gamma_T, mu_T_x
    #     ).squeeze()  # Remove inflate, only keep batch dimension for x-part.
    #
    #     sigma_T_h = self.sigma(gamma_T, mu_T_h)
    #
    #     # Compute KL for h-part.
    #     zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
    #     kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros, ones, node_mask)
    #
    #     # Compute KL for x-part.
    #     zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
    #     subspace_d = self.subspace_dimensionality(node_mask)
    #     kl_distance_x = gaussian_KL_for_dimension(
    #         mu_T_x, sigma_T_x, zeros, ones, d=subspace_d
    #     )
    #
    #     return kl_distance_x + kl_distance_h

    def compute_x_pred(self, net_out, zt, gamma_t):
        """Commputes x_pred, i.e. the most likely prediction of x."""

        sigma_t = self.sigma(gamma_t, target_tensor=net_out)
        alpha_t = self.alpha(gamma_t, target_tensor=net_out)
        eps_t = net_out
        x_pred = 1.0 / alpha_t * (zt - sigma_t * eps_t)

        return x_pred

    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context,
                             # fix_noise=False
                             ):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.snr(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.sample_normal(
            mu=mu_x, sigma=sigma_x, node_mask=node_mask,
            # fix_noise=fix_noise
        )

        x = xh[:, :, : self.n_dims]

        x, h_cat = self.unnormalize(x, z0[:, :, self.n_dims : -1], node_mask)

        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        h = h_cat
        return x, h

    def sample_normal(self, mu, sigma, node_mask,
                      # fix_noise=False
                      ):
        """Samples from a Normal distribution."""
        # bs = 1 if fix_noise else mu.size(0)
        bs = mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps

    def sample_p_zs_given_zt(
        self, s: torch.Tensor, t: torch.Tensor, zt: torch.Tensor, node_mask: torch.Tensor, edge_mask: torch.Tensor, context: torch.Tensor,
            # fix_noise=False
    ):
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

        # Compute mu for p(zs | zt).
        # Redundant checks removed
        # self.assert_mean_zero_with_mask(zt[:, :, : self.n_dims], node_mask)
        # self.assert_mean_zero_with_mask(eps_t[:, :, : self.n_dims], node_mask)

        mu = (
            zt / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        )

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask)
                                # fix_noise)

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [
                remove_mean_with_mask(zs[:, :, : self.n_dims], node_mask),
                zs[:, :, self.n_dims :],
            ],
            dim=2,
        )
        return zs

    def sample_combined_position_feature_noise(self, n_samples: int, n_nodes: int, node_mask: torch.Tensor):
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

    # Renamed from sample to allow compilation with torch.jit.script
    def forward(
        self, n_samples: int, n_nodes: int, node_mask: torch.Tensor, edge_mask: torch.Tensor, context: torch.Tensor,
            # fix_noise=False
    ):
        """
        Draw samples from the generative model.
        Inference
        """
        # if fix_noise:
        #     # Noise is broadcasted over the batch axis, useful for visualizations.
        #     z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        # else:

        # Handle case of single sample generation? due to optimisations in flow of EGNN Dynamics

        z = self.sample_combined_position_feature_noise(
                n_samples, n_nodes, node_mask
            )

        # Remove redundant checks
        # self.assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)


        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        # for s in reversed(range(0, self.T)):
        for s in self.timesteps:
            s_array = torch.full([n_samples, 1], fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s_array, t_array, z, node_mask, edge_mask, context,
                # fix_noise=fix_noise
            )

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(
            z, node_mask, edge_mask, context,
            # fix_noise=fix_noise
        )

        # Remove redundant checks
        # self.assert_mean_zero_with_mask(x, node_mask)

        # max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        # if max_cog > torch.tensor(5e-2):
        #     # print(
        #     #     f"Warning cog drift with error {max_cog:.3f}. Projecting the positions down."
        #     # )
        #     x = remove_mean_with_mask(x, node_mask)

        return x, h

    # @torch.no_grad()
    # def sample_chain(
    #     self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None
    # ):
    #     """
    #     Draw samples from the generative model, keep the intermediate states for visualization purposes.
    #     """
    #     z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)
    #
    #     assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)
    #
    #     if keep_frames is None:
    #         keep_frames = self.T
    #     else:
    #         assert keep_frames <= self.T
    #     chain = torch.zeros((keep_frames,) + z.size(), device=z.device)
    #
    #     # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
    #     for s in reversed(range(0, self.T)):
    #         s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
    #         t_array = s_array + 1
    #         s_array = s_array / self.T
    #         t_array = t_array / self.T
    #
    #         z = self.sample_p_zs_given_zt(
    #             s_array, t_array, z, node_mask, edge_mask, context
    #         )
    #
    #         assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)
    #
    #         # Write to chain tensor.
    #         write_index = (s * keep_frames) // self.T
    #         chain[write_index] = self.unnormalize_z(z, node_mask)
    #
    #     # Finally sample p(x, h | z_0).
    #     x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)
    #
    #     assert_mean_zero_with_mask(x[:, :, : self.n_dims], node_mask)
    #
    #     xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
    #     chain[0] = xh  # Overwrite last frame with the resulting x and h.
    #
    #     chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])
    #
    #     return chain_flat
