import torch
from torch import Tensor

from .utils import sample_combined_position_feature_noise, remove_mean_with_mask
from .egnn import EGNNDynamics
from typing import Tuple

class EquivariantFlowMatching(torch.nn.Module):
    def __init__(self,
     dynamics: EGNNDynamics,
     in_node_nf: int = 8,
     n_dims: int = 3,
     timesteps: int = 1000,
     noise_precision: float = 1e-4,
     norm_values: Tuple[float, float] = (1.0, 9.0),
     ):
        super().__init__()
        self.dynamics = dynamics
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.timesteps = timesteps
        self.noise_precision = noise_precision
        self.norm_values = norm_values

    def velocity(self, 
                xh: Tensor,
                t: Tensor,
                node_mask: Tensor,
                edge_mask: Tensor,
                context: Tensor,
    ) -> Tensor:
        """
        EGNN Dynamics Model forward pass to compute the velocity field v(x, t)
        """
        net_out = self.dynamics(t, xh, node_mask, edge_mask, context)
        return net_out

    def sample_combined_position_feature_noise(self,
        n_samples: int,
        n_nodes: int,
        node_mask: Tensor,
    ) -> Tensor:
        """
        Samples combined position and feature noise for the input noise x0
        """
        return sample_combined_position_feature_noise(
            n_samples, n_nodes, node_mask, self.in_node_nf, self.n_dims
        )

    def decode(self, z: Tensor, node_mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Latent z → (x, h) like EDM sample_p_xh_given_z0, without phi / noise.
        """
        x = z[:, :, : self.n_dims]
        h_cat = z[:, :, self.n_dims : -1]  # same slice as EDM
        # unnormalize (same as EquivariantDiffusion.unnormalize)
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] * node_mask
        h = torch.nn.functional.one_hot(
            torch.argmax(h_cat, dim=2),
            num_classes=self.in_node_nf,  # 8 classes; confirm vs n_dims:-1 width
        ).float() * node_mask
        x = remove_mean_with_mask(x, node_mask)  # optional; EDM often COM-free already
        return x, h
    
    def compute_loss(self, 
                    x0: Tensor, # Input Noise zT
                    x1: Tensor, # Final Molecule z0
                    node_mask: Tensor,
                    edge_mask: Tensor,
                    context: Tensor
                ) -> Tensor:
        
        t = torch.rand(x0.shape[0], 1, device=x0.device)
        xt = (1 - t[:, None, :]) * x0 + t[:, None, :] * x1
        xt = torch.cat([remove_mean_with_mask(xt[..., :3], node_mask), xt[..., 3:]], -1) * node_mask
        target = (x1 - x0) * node_mask
        pred = self.velocity(xt, t, node_mask, edge_mask, context)
        # Masked MSE loss
        loss = ((pred - target) ** 2 * node_mask).sum() / node_mask.sum().clamp_min(1)
        return loss

    def step(self,
             xt: Tensor,
            t_start: torch.Tensor,
            t_end: float,
            node_mask: Tensor,
            edge_mask: Tensor,
            context: Tensor) -> Tensor:
        """
        Simple midpoint step rule for the velocity field integration.
        """
        dt = t_end - t_start
        t_mid = t_start + dt / 2
        k1 = self.velocity(t_start, xt, ...)
        x_mid = xt + k1 * (dt / 2)
        x_mid[..., :3] = remove_mean_with_mask(x_mid[..., :3], node_mask)
        v_mid = self.velocity(t_mid, x_mid, ...)
        out = xt + dt * v_mid
        out[..., :3] = remove_mean_with_mask(out[..., :3], node_mask)
        return out * node_mask

    def sample(self, 
               node_mask: Tensor,
               edge_mask: Tensor,
               context: Tensor,
               n_steps: int=50):

        """
        Samples a molecule from teh generative model using flow matching algorithm.
        """
        z = self.sample_combined_position_feature_noise(...)
        ts = torch.linspace(0, 1, n_steps + 1, device=...)
        for i in range(n_steps):
            z = self.step(z, ts[i], ts[i+1], node_mask, edge_mask, context)
        # decode h with argmax like sample_p_xh_given_z0 (without gamma)
        x, h = self.decode(z, node_mask)
        return x, h

  