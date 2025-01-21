import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

from .egnn import EGNNDynamics



def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


def cdf_standard_gaussian(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


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


def assert_mean_zero_with_mask(x, node_mask, eps=1e-10):
    assert_correctly_masked(x, node_mask)
    largest_value = x.abs().max().item()
    error = torch.sum(x, dim=1, keepdim=True).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < 1e-2, f"Mean is not zero, relative_error {rel_error}"


def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f"Error {masked_max_abs_value} too high"
    N = node_mask.sum(1, keepdims=True)

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
    # print(f"q_sigma - {q_sigma.size()}")
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return (
        d * torch.log(p_sigma / q_sigma)
        + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2)
        - 0.5 * d
    )


def sample_center_gravity_zero_gaussian_with_mask(size, device, node_mask):
    assert len(size) == 3
    x = torch.randn(size, device=device)

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def sample_gaussian_with_mask(size, device, node_mask):
    x = torch.randn(size, device=device)

    x_masked = x * node_mask
    return x_masked


class PositiveLinear(torch.nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init_offset: int = -2,
    ):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = F.softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, timesteps, precision, power: int = 2):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        # Default Schedule - polynomial with power 2

        alphas2 = polynomial_schedule(timesteps, s=precision, power=power)

        # print("alphas2", alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = torch.log(alphas2)
        log_sigmas2 = torch.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        # print("gamma", -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            (-log_alphas2_to_sigmas2).float(), requires_grad=False
        )

    def forward(self, t):
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
        noise_precision=1e-4,
        norm_values=(
            1.0,
            9.0,
        ),  # -> Original (1, 8, 1) - (1, max number of atom classes, used for h_integer)
        norm_biases=(None, 0.0),
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

        self.T = timesteps

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer("buffer", torch.zeros(1))

        self.check_issues_norm_values()

    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        max_norm_value = self.norm_values[1]

        if sigma_0 * num_stdevs > 1.0 / max_norm_value:
            raise ValueError(
                f"Value for normalization value {max_norm_value} probably too "
                f"large with sigma_0 {sigma_0:.5f} and "
                f"1 / norm_value = {1. / max_norm_value}"
            )

    def phi(self, x, t, node_mask, edge_mask, context):
        net_out = self.dynamics._forward(t, x, node_mask, edge_mask, context)

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

    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims

    def normalize(self, x, h, node_mask):
        x = x / self.norm_values[0]
        delta_log_px = -self.subspace_dimensionality(node_mask) * math.log(
            self.norm_values[0]
        )

        # Casting to float in case h still has long or int type.
        h = (h.float() - self.norm_biases[1]) / self.norm_values[1] * node_mask

        return x, h, delta_log_px

    def unnormalize(self, x, h_cat, node_mask):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask

        return x, h_cat

    # def unnormalize_z(self, z, node_mask):
    #     # Parse from z
    #     x, h_cat = (
    #         z[:, :, 0 : self.n_dims],
    #         z[:, :, self.n_dims : self.n_dims + self.num_classes],
    #     )
    #     h_int = z[
    #         :, :, self.n_dims + self.num_classes : self.n_dims + self.num_classes + 1
    #     ]
    #     assert h_int.size(2) == self.include_charges
    #
    #     # Unnormalize
    #     x, h_cat, h_int = self.unnormalize(x, h_cat, h_int, node_mask)
    #     output = torch.cat([x, h_cat, h_int], dim=2)
    #     return output

    def sigma_and_alpha_t_given_s(
        self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor
    ):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)), target_tensor
        )

        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def kl_prior(self, xh, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((xh.size(0), 1), device=xh.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh)

        # Compute means.
        mu_T = alpha_T * xh
        # print(f"muT - {mu_T}")
        mu_T_x, mu_T_h = mu_T[:, :, : self.n_dims], mu_T[:, :, self.n_dims :]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(
            gamma_T, mu_T_x
        ).squeeze()  # Remove inflate, only keep batch dimension for x-part.

        # print(f"sigma_T_x - {sigma_T_x.size()}")
        sigma_T_h = self.sigma(gamma_T, mu_T_h)

        # Compute KL for h-part.
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros, ones, node_mask)

        # Compute KL for x-part.
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        subspace_d = self.subspace_dimensionality(node_mask)
        kl_distance_x = gaussian_KL_for_dimension(
            mu_T_x, sigma_T_x, zeros, ones, d=subspace_d
        )

        return kl_distance_x + kl_distance_h

    def compute_x_pred(self, net_out, zt, gamma_t):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        # if self.parametrization == "x":
        #     x_pred = net_out
        # elif self.parametrization == "eps":
        sigma_t = self.sigma(gamma_t, target_tensor=net_out)
        alpha_t = self.alpha(gamma_t, target_tensor=net_out)
        eps_t = net_out
        x_pred = 1.0 / alpha_t * (zt - sigma_t * eps_t)
        # else:
        #     raise ValueError(self.parametrization)

        return x_pred

    # def compute_error(self, net_out, eps):
    #     """Computes error, i.e. the most likely prediction of x."""
    #     eps_t = net_out
    #
    #     if self.training:
    #         denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
    #         error = sum_except_batch((eps - eps_t) ** 2) / denom
    #     else:
    #         error = sum_except_batch((eps - eps_t) ** 2)
    #
    #     # if self.training and self.loss_type == "l2":
    #     #     denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
    #     #     error = sum_except_batch((eps - eps_t) ** 2) / denom
    #     # else:
    #     #     error = sum_except_batch((eps - eps_t) ** 2)
    #     return error

    # def log_constants_p_x_given_z0(self, x, node_mask):
    #     """Computes p(x|z0)."""
    #     batch_size = x.size(0)
    #
    #     n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
    #     assert n_nodes.size() == (batch_size,)
    #     degrees_of_freedom_x = (n_nodes - 1) * self.n_dims
    #
    #     zeros = torch.zeros((x.size(0), 1), device=x.device)
    #     gamma_0 = self.gamma(zeros)
    #
    #     # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
    #     log_sigma_x = 0.5 * gamma_0.view(batch_size)
    #
    #     return degrees_of_freedom_x * (-log_sigma_x - 0.5 * math.log(2 * math.pi))

    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, context, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.snr(-0.5 * gamma_0).unsqueeze(1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.sample_normal(
            mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise
        )

        x = xh[:, :, : self.n_dims]

        # h_int = z0[:, :, -1:] if self.include_charges else torch.zeros(0).to(z0.device)
        x, h_cat = self.unnormalize(x, z0[:, :, self.n_dims : -1], node_mask)

        h_cat = F.one_hot(torch.argmax(h_cat, dim=2), self.num_classes) * node_mask
        # h_int = torch.round(h_int).long() * node_mask
        h = h_cat
        return x, h

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps

    # def log_pxh_given_z0_without_constants(
    #     self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10
    # ):
    #     # Discrete properties are predicted directly from z_t.
    #
    #     z_h_cat = z_t[:, :, self.n_dims :]
    #
    #     # Take only part over x.
    #     eps_x = eps[:, :, : self.n_dims]
    #     net_x = net_out[:, :, : self.n_dims]
    #
    #     # Compute sigma_0 and rescale to the integer scale of the data.
    #     sigma_0 = self.sigma(gamma_0, target_tensor=z_t)
    #     sigma_0_cat = sigma_0 * self.norm_values[1]
    #
    #     # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
    #     # the weighting in the epsilon parametrization is exactly '1'.
    #     log_p_x_given_z_without_constants = -0.5 * self.compute_error(net_x, eps_x)
    #
    #     onehot = h * self.norm_values[1] + self.norm_biases[1]
    #
    #     estimated_h_cat = z_h_cat * self.norm_values[1] + self.norm_biases[1]
    #
    #     # Centered h_cat around 1, since onehot encoded.
    #     centered_h_cat = estimated_h_cat - 1
    #
    #     # Compute integrals from 0.5 to 1.5 of the normal distribution
    #     # N(mean=z_h_cat, stdev=sigma_0_cat)
    #     log_ph_cat_proportional = torch.log(
    #         cdf_standard_gaussian((centered_h_cat + 0.5) / sigma_0_cat)
    #         - cdf_standard_gaussian((centered_h_cat - 0.5) / sigma_0_cat)
    #         + epsilon
    #     )
    #
    #     # Normalize the distribution over the categories.
    #     log_Z = torch.logsumexp(log_ph_cat_proportional, dim=2, keepdim=True)
    #     log_probabilities = log_ph_cat_proportional - log_Z
    #
    #     # Select the log_prob of the current category usign the onehot
    #     # representation.
    #
    #     log_ph_cat = sum_except_batch(log_probabilities * onehot * node_mask)
    #
    #     # Combine categorical and integer log-probabilities.
    #     log_p_h_given_z = log_ph_cat  # + log_ph_integer
    #
    #     # Combine log probabilities for x and h.
    #     log_p_xh_given_z = log_p_x_given_z_without_constants + log_p_h_given_z
    #
    #     return log_p_xh_given_z

    # def compute_loss(self, x, h, node_mask, edge_mask, context, t0_always):
    #     """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""
    #
    #     # This part is about whether to include loss term 0 always.
    #     if t0_always:
    #         # loss_term_0 will be computed separately.
    #         # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
    #         lowest_t = 1
    #     else:
    #         # estimator = loss_t,           where t ~ U({0, ..., T})
    #         lowest_t = 0
    #
    #     # Sample a timestep t.
    #     t_int = torch.randint(
    #         lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device
    #     ).float()
    #     s_int = t_int - 1
    #     t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).
    #
    #     # Normalize t to [0, 1]. Note that the negative
    #     # step of s will never be used, since then p(x | z0) is computed.
    #     s = s_int / self.T
    #     t = t_int / self.T
    #
    #     # Compute gamma_s and gamma_t via the network.
    #     gamma_s = self.inflate_batch_array(self.gamma(s), x)
    #     gamma_t = self.inflate_batch_array(self.gamma(t), x)
    #
    #     # Compute alpha_t and sigma_t from gamma.
    #     alpha_t = self.alpha(gamma_t, x)
    #     sigma_t = self.sigma(gamma_t, x)
    #
    #     # Sample zt ~ Normal(alpha_t x, sigma_t)
    #     eps = self.sample_combined_position_feature_noise(
    #         n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask
    #     )
    #
    #     # Concatenate x, h[integer] and h[categorical].
    #
    #     xh = torch.cat([x, h], dim=2)
    #
    #     # Sample z_t given x, h for timestep t, from q(z_t | x, h)
    #
    #     z_t = alpha_t * xh + sigma_t * eps
    #
    #     assert_mean_zero_with_mask(z_t[:, :, : self.n_dims], node_mask)
    #
    #     # Neural net prediction.
    #     net_out = self.phi(z_t, t, node_mask, edge_mask, context)
    #
    #     # Compute the error.
    #     error = self.compute_error(net_out, eps)
    #
    #     if self.training:
    #         SNR_weight = torch.ones_like(error)
    #     else:
    #         # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
    #         SNR_weight = (self.snr(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
    #
    #     assert error.size() == SNR_weight.size()
    #     loss_t_larger_than_zero = 0.5 * SNR_weight * error
    #
    #     # The _constants_ depending on sigma_0 from the
    #     # cross entropy term E_q(z0 | x) [log p(x | z0)].
    #     neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask)
    #
    #     # Reset constants during training with l2 loss.
    #
    #     if self.training:
    #         neg_log_constants = torch.zeros_like(neg_log_constants)
    #
    #     # if self.training and self.loss_type == "l2":
    #     #     neg_log_constants = torch.zeros_like(neg_log_constants)
    #
    #     # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
    #     kl_prior = self.kl_prior(xh, node_mask)
    #
    #     # Combining the terms
    #     if t0_always:
    #         loss_t = loss_t_larger_than_zero
    #         num_terms = self.T  # Since t=0 is not included here.
    #         estimator_loss_terms = num_terms * loss_t
    #
    #         # Compute noise values for t = 0.
    #         t_zeros = torch.zeros_like(s)
    #         gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
    #         alpha_0 = self.alpha(gamma_0, x)
    #         sigma_0 = self.sigma(gamma_0, x)
    #
    #         # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
    #         eps_0 = self.sample_combined_position_feature_noise(
    #             n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask
    #         )
    #         z_0 = alpha_0 * xh + sigma_0 * eps_0
    #
    #         net_out = self.phi(z_0, t_zeros, node_mask, edge_mask, context)
    #
    #         loss_term_0 = -self.log_pxh_given_z0_without_constants(
    #             x, h, z_0, gamma_0, eps_0, net_out, node_mask
    #         )
    #
    #         assert kl_prior.size() == estimator_loss_terms.size()
    #         assert kl_prior.size() == neg_log_constants.size()
    #         assert kl_prior.size() == loss_term_0.size()
    #
    #         loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0
    #
    #     else:
    #         # Computes the L_0 term (even if gamma_t is not actually gamma_0)
    #         # and this will later be selected via masking.
    #         loss_term_0 = -self.log_pxh_given_z0_without_constants(
    #             x, h, z_t, gamma_t, eps, net_out, node_mask
    #         )
    #
    #         t_is_not_zero = 1 - t_is_zero
    #
    #         loss_t = (
    #             loss_term_0 * t_is_zero.squeeze()
    #             + t_is_not_zero.squeeze() * loss_t_larger_than_zero
    #         )
    #
    #         # Only upweigh estimator if using the vlb objective.
    #
    #         if self.training:
    #             estimator_loss_terms = loss_t
    #         else:
    #             num_terms = self.T + 1  # Includes t = 0.
    #             estimator_loss_terms = num_terms * loss_t
    #
    #         assert kl_prior.size() == estimator_loss_terms.size()
    #         assert kl_prior.size() == neg_log_constants.size()
    #
    #         loss = kl_prior + estimator_loss_terms + neg_log_constants
    #
    #     assert len(loss.shape) == 1, f"{loss.shape} has more than only batch dim."
    #
    #     return loss, {
    #         "t": t_int.squeeze(),
    #         "loss_t": loss.squeeze(),
    #         "error": error.squeeze(),
    #     }

    # def forward(self, x, h, node_mask=None, edge_mask=None, context=None):
    #     """
    #     Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
    #     """
    #     # Normalize data, take into account volume change in x.
    #
    #     x, h, delta_log_px = self.normalize(x, h, node_mask)
    #
    #     # Reset delta_log_px if not vlb objective.
    #     if self.training:
    #         delta_log_px = torch.zeros_like(delta_log_px)
    #
    #     if self.training:
    #         # Only 1 forward pass when t0_always is False.
    #         loss, loss_dict = self.compute_loss(
    #             x, h, node_mask, edge_mask, context, t0_always=False
    #         )
    #     else:
    #         # Less variance in the estimator, costs two forward passes.
    #         loss, loss_dict = self.compute_loss(
    #             x, h, node_mask, edge_mask, context, t0_always=True
    #         )
    #
    #     neg_log_pxh = loss
    #
    #     # Correct for normalization on x.
    #     assert neg_log_pxh.size() == delta_log_px.size()
    #     neg_log_pxh = neg_log_pxh - delta_log_px
    #
    #     return neg_log_pxh

    def sample_p_zs_given_zt(
        self, s, t, zt, node_mask, edge_mask, context, fix_noise=False
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
        assert_mean_zero_with_mask(zt[:, :, : self.n_dims], node_mask)
        assert_mean_zero_with_mask(eps_t[:, :, : self.n_dims], node_mask)
        mu = (
            zt / alpha_t_given_s
            - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t
        )

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

        # Project down to avoid numerical runaway of the center of gravity.
        zs = torch.cat(
            [
                remove_mean_with_mask(zs[:, :, : self.n_dims], node_mask),
                zs[:, :, self.n_dims :],
            ],
            dim=2,
        )
        return zs

    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
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
            ),  # -> initial, most likely a typo as size shoud match node_mask
            # size=(n_samples, n_nodes, self.n_dims),  # changed self.in_nodes to self.n_dims ???
            device=node_mask.device,
            node_mask=node_mask,
        )
        z = torch.cat([z_x, z_h], dim=2)
        # print(f"sample combined pos feature noise z - {z.size()}")
        return z

    @torch.no_grad()
    def sample(
        self, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False
    ):
        """
        Draw samples from the generative model.
        Inference
        """
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
        else:
            z = self.sample_combined_position_feature_noise(
                n_samples, n_nodes, node_mask
            )

        assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in tqdm(reversed(range(0, self.T))):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s_array, t_array, z, node_mask, edge_mask, context, fix_noise=fix_noise
            )

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(
            z, node_mask, edge_mask, context, fix_noise=fix_noise
        )

        assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(
                f"Warning cog drift with error {max_cog:.3f}. Projecting "
                f"the positions down."
            )
            x = remove_mean_with_mask(x, node_mask)

        return x, h

    @torch.no_grad()
    def sample_chain(
        self, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None
    ):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z.size(), device=z.device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z = self.sample_p_zs_given_zt(
                s_array, t_array, z, node_mask, edge_mask, context
            )

            assert_mean_zero_with_mask(z[:, :, : self.n_dims], node_mask)

            # Write to chain tensor.
            write_index = (s * keep_frames) // self.T
            chain[write_index] = self.unnormalize_z(z, node_mask)

        # Finally sample p(x, h | z_0).
        x, h = self.sample_p_xh_given_z0(z, node_mask, edge_mask, context)

        assert_mean_zero_with_mask(x[:, :, : self.n_dims], node_mask)

        xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
        chain[0] = xh  # Overwrite last frame with the resulting x and h.

        chain_flat = chain.view(n_samples * keep_frames, *z.size()[1:])

        return chain_flat

    # def log_info(self):
    #     """
    #     Some info logging of the model.
    #     """
    #     gamma_0 = self.gamma(torch.zeros(1, device=self.buffer.device))
    #     gamma_1 = self.gamma(torch.ones(1, device=self.buffer.device))
    #
    #     log_snr_max = -gamma_0
    #     log_snr_min = -gamma_1
    #
    #     info = {"log_SNR_max": log_snr_max.item(), "log_SNR_min": log_snr_min.item()}
    #
    #     return info
