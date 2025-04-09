from typing import Tuple, Union

import numpy as np


def clip_noise_schedule(alphas2: np.ndarray, clip_value: float = 0.001) -> np.ndarray:
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """

    alphas2 = np.concatenate((np.ones(1), alphas2), axis=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def polynomial_schedule(timesteps: int, s: float = 1e-4, power: int = 2):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.

    Remark - rewritten in torch only
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def remove_mean_with_mask(x, node_mask):
    n = np.sum(node_mask, 1, keepdims=True)

    mean = np.sum(x, 1, keepdims=True) / n
    x = x - mean * node_mask
    return x


def sample_center_gravity_zero_gaussian_with_mask(
    size: Tuple[int, int, int], node_mask: np.ndarray
):
    x = np.random.rand(size[0], size[1], size[2])

    x_masked = x * node_mask

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    return x_projected


def sample_gaussian_with_mask(size: Tuple[int, int, int], node_mask: np.ndarray):
    x = np.random.randn(size[0], size[1], size[2])

    x_masked = x * node_mask
    return x_masked


class PredefinedNoiseSchedule:
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, timesteps: int, precision: float, power: int = 2):
        self.timesteps = timesteps

        # Default Schedule - polynomial with power 2

        alphas2 = polynomial_schedule(timesteps, s=precision, power=power)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        self.gamma = (-log_alphas2_to_sigmas2).astype(np.float32)

    def __call__(self, t: np.ndarray):
        t_int = np.round(t * self.timesteps).astype(int)
        return self.gamma[t_int]


class EquivariantDiffusionONNX:
    """
    PyTorch - free implementation of
    The E(n) Diffusion Module.
    """

    def __init__(
        self,
        egnn_onnx: str,
        in_node_nf: int = 8,
        n_dims: int = 3,
        timesteps: int = 1000,
        noise_precision: float = 1e-4,
        norm_values: Tuple[float, float] = (
            1.0,
            9.0,
        ),  # (1, max number of atom classes)
    ):
        try:
            import onnxruntime
        except ImportError as e:
            raise ImportError(
                'Failed to import onnxruntime. To resolve run `pip install "mlconfgen[onnx]"`\n'
            ) from e

        super().__init__()

        self.gamma = PredefinedNoiseSchedule(
            timesteps=timesteps, precision=noise_precision
        )

        # The network that will predict the denoising.

        self.dynamics = onnxruntime.InferenceSession(egnn_onnx)

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.num_classes = self.in_node_nf

        # Declare time steps-related tensors
        self.T = timesteps
        self.time_steps = np.flip(np.arange(0, timesteps))

        self.norm_values = norm_values

    def phi(self, x, t, node_mask, edge_mask, context):
        inputs = {
            "t": t.astype(np.float32),
            "xh": x.astype(np.float32),
            "node_mask": node_mask.astype(np.float32),
            "edge_mask": edge_mask.astype(np.float32),
            "context": context.astype(np.float32),
        }

        net_out = self.dynamics.run(None, inputs)
        return net_out[0]

    @staticmethod
    def sigmoid(z: Union[np.ndarray, float]):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def softplus(z: Union[np.ndarray, float]):
        return np.log1p(np.exp(z))

    @staticmethod
    def logsigmoid(z: Union[np.ndarray, float]):
        return -np.log1p(np.exp(-z))

    @staticmethod
    def one_hot(labels: np.ndarray, num_classes: int):
        return np.eye(num_classes)[labels]

    @staticmethod
    def inflate_batch_array(array: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.shape[0],) + (1,) * (len(target.shape) - 1)
        return array.reshape(target_shape)

    def sigma(self, gamma: np.ndarray, target_tensor: np.ndarray) -> np.ndarray:
        """Computes sigma given gamma."""
        return self.inflate_batch_array(np.sqrt(self.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma: np.ndarray, target_tensor: np.ndarray):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(np.sqrt(self.sigmoid(-gamma)), target_tensor)

    @staticmethod
    def snr(gamma: np.ndarray) -> np.ndarray:
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return np.exp(-gamma)

    def unnormalize(
        self, x: np.ndarray, h_cat: np.ndarray, node_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1]

        h_cat = h_cat * node_mask

        return x, h_cat

    def sigma_and_alpha_t_given_s(
        self, gamma_t: np.ndarray, gamma_s: np.ndarray, target_tensor: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            1 - np.exp(self.softplus(gamma_s) - self.softplus(gamma_t)), target_tensor
        )

        log_alpha2_t = self.logsigmoid(-gamma_t)
        log_alpha2_s = self.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = np.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(alpha_t_given_s, target_tensor)

        sigma_t_given_s = np.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def compute_x_pred(
        self, net_out: np.ndarray, zt: np.ndarray, gamma_t: np.ndarray
    ) -> np.ndarray:
        """Commputes x_pred, i.e. the most likely prediction of x."""

        sigma_t = self.sigma(gamma_t, target_tensor=net_out)
        alpha_t = self.alpha(gamma_t, target_tensor=net_out)
        eps_t = net_out
        x_pred = 1.0 / alpha_t * (zt - sigma_t * eps_t)

        return x_pred

    def sample_p_xh_given_z0(
        self,
        z0: np.ndarray,
        node_mask: np.ndarray,
        edge_mask: np.ndarray,
        context: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Samples x ~ p(x|z0)."""
        zeros = np.zeros(shape=(z0.shape[0], 1))
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = np.expand_dims(self.snr(-0.5 * gamma_0), 1)
        net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask)

        x = xh[:, :, : self.n_dims]

        x, h_cat = self.unnormalize(x, z0[:, :, self.n_dims : -1], node_mask)

        h_cat = self.one_hot(np.argmax(h_cat, axis=2), self.num_classes) * node_mask
        h = h_cat
        return x, h

    def sample_normal(
        self, mu: np.ndarray, sigma: np.ndarray, node_mask: np.ndarray
    ) -> np.ndarray:
        """Samples from a Normal distribution."""
        bs = mu.shape[0]
        eps = self.sample_combined_position_feature_noise(bs, mu.shape[1], node_mask)
        return mu + sigma * eps

    def sample_p_zs_given_zt(
        self,
        s: np.ndarray,
        t: np.ndarray,
        zt: np.ndarray,
        node_mask: np.ndarray,
        edge_mask: np.ndarray,
        context: np.ndarray,
    ) -> np.ndarray:
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

        # Sample zs given the parameters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask)

        # Project down to avoid numerical runaway of the center of gravity.
        zs = np.concatenate(
            [
                remove_mean_with_mask(zs[:, :, : self.n_dims], node_mask),
                zs[:, :, self.n_dims :],
            ],
            axis=2,
        )
        return zs

    def sample_combined_position_feature_noise(
        self, n_samples: int, n_nodes: int, node_mask: np.ndarray
    ) -> np.ndarray:
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims),
            node_mask=node_mask,
        )

        z_h = sample_gaussian_with_mask(
            size=(
                n_samples,
                n_nodes,
                self.in_node_nf,
            ),
            node_mask=node_mask,
        )

        z = np.concatenate([z_x, z_h], axis=2)
        return z

    def __call__(
        self,
        node_mask: np.ndarray,
        edge_mask: np.ndarray,
        context: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Draw samples from the generative model.
        Inference
        """
        n_samples, n_nodes, _ = node_mask.shape

        z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in self.time_steps:
            s_array = np.full([n_samples, 1], fill_value=s)
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
