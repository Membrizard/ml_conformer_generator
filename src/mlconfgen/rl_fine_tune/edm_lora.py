import torch
import torch.nn as nn

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal
from ..equivariant_diffusion import sample_center_gravity_zero_gaussian_with_mask, sample_gaussian_with_mask


class EDMLoRAPolicy(nn.Module):
    def __init__(
        self,
        device,
        h_dim: int = 8,
        normalization_factor: int = 100,
        coords_range: float = 15.0,
        edges_in_d: int = 2,
        sigma_x: float = 0.05,
        sigma_h: float = 0.05,
        x_scale: float = 0.25,
        h_scale: float = 0.25,
    ):
        super().__init__()

        self.h_dim = h_dim
        self.device = device

        self.sigma_x = sigma_x
        self.sigma_h = sigma_h

        # scales limit how much the adapter can change the EDM outputs
        self.x_scale = x_scale
        self.h_scale = h_scale

        self.x_update = EquivariantUpdate(
            hidden_nf=h_dim,
            normalization_factor=normalization_factor,
            edges_in_d=edges_in_d,
            coords_range=coords_range,
        )
        self.h_update = GCL(
            input_nf=h_dim,
            output_nf=h_dim,
            hidden_nf=h_dim,
            normalization_factor=normalization_factor,
            edges_in_d=edges_in_d,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        edge_mask: torch.Tensor,
        node_mask: torch.Tensor,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """
        x: [B, N, 3]
        h: [B, N, H]
        edge_mask: expected by your GCL / EquivariantUpdate
        node_mask: expected by your GCL / EquivariantUpdate

        Returns:
            x_new: [B, N, 3]
            h_new: [B, N, H]
            log_prob: [B]
            aux: dict with useful regularizers/statistics
        """
        bs, n_nodes, _ = x.size()

        x_in = x
        h_in = h

        x_flat = x.view(bs * n_nodes, 3)
        h_flat = h.view(bs * n_nodes, self.h_dim)
        node_mask_flat = node_mask.view(bs * n_nodes, 1)

        edge_index = get_adj_matrix(n_nodes, bs, self.device)

        distances, coord_diff = coord2diff(x_flat, edge_index)
        edge_attr = torch.cat([distances, distances], dim=1)

        dh_mean_flat = self.h_update(
            h=h_flat,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_mask=node_mask_flat,
            edge_mask=edge_mask,
        )

        dx_mean_flat = self.x_update(
            h=dh_mean_flat,
            coord=x_flat,
            edge_index=edge_index,
            coord_diff=coord_diff,
            edge_attr=edge_attr,
            node_mask=node_mask_flat,
            edge_mask=edge_mask,
        )

        # Interpret outputs as small residual means, not full replacements
        dx_mean = dx_mean_flat.view(bs, n_nodes, 3)
        dh_mean = dh_mean_flat.view(bs, n_nodes, self.h_dim)

        dx_mean = self.x_scale * dx_mean
        dh_mean = self.h_scale * dh_mean

        if sample:
            # eps_x = torch.randn_like(dx_mean)
            # eps_h = torch.randn_like(dh_mean)

            eps_x = sample_center_gravity_zero_gaussian_with_mask(dx_mean.size(), self.device, node_mask)
            eps_h = sample_gaussian_with_mask(dh_mean.size(), self.device, node_mask)

            dx = dx_mean + self.sigma_x * eps_x
            dh = dh_mean + self.sigma_h * eps_h
        else:
            dx = dx_mean
            dh = dh_mean

        x_new = x_in + dx
        h_new = h_in + dh

        # Gaussian log-prob of sampled residuals under the adapter policy
        dist_x = Normal(
            loc=dx_mean,
            scale=torch.full_like(dx_mean, self.sigma_x),
        )
        dist_h = Normal(
            loc=dh_mean,
            scale=torch.full_like(dh_mean, self.sigma_h),
        )

        # Sum over nodes/features -> [B]
        log_prob_x = dist_x.log_prob(dx).sum(dim=(-1, -2))
        log_prob_h = dist_h.log_prob(dh).sum(dim=(-1, -2))
        log_prob = log_prob_x + log_prob_h

        aux = {
            "dx_mean_l2": dx_mean.pow(2).mean(),
            "dh_mean_l2": dh_mean.pow(2).mean(),
            "dx_sample_l2": dx.pow(2).mean(),
            "dh_sample_l2": dh.pow(2).mean(),
            "log_prob_x_mean": log_prob_x.mean(),
            "log_prob_h_mean": log_prob_h.mean(),
        }

        return x_new, h_new, log_prob, aux




# class EDMLoRA(nn.Module):
#     def __init__(self, device):
#         super(EDMLoRA, self).__init__()
#
#         edges_in_d = 2
#         nf = 100
#         coords_range = 15
#         n_hidden = 8
#
#         self.x_update = EquivariantUpdate(
#             hidden_nf=n_hidden,
#             normalization_factor=nf,
#             edges_in_d=edges_in_d,
#             coords_range=coords_range,
#         )
#         self.h_update = GCL(
#             input_nf=n_hidden,
#             output_nf=n_hidden,
#             hidden_nf=n_hidden,
#             normalization_factor=nf,
#             edges_in_d=edges_in_d,
#         )
#         self.device = device
#
#     def forward(self, x, h, edge_mask, node_mask):
#         bs, n_nodes, _ = x.size()
#         x = x.view(bs * n_nodes, 3)
#         h = h.view(bs * n_nodes, 8)
#
#         edge_index = get_adj_matrix(n_nodes, bs, self.device)
#
#         distances, coord_diff = coord2diff(x, edge_index)
#         edge_attr = torch.cat([distances, distances], dim=1)
#
#         h = self.h_update(
#             h=h,
#             edge_index=edge_index,
#             edge_attr=edge_attr,
#             node_mask=node_mask,
#             edge_mask=edge_mask,
#         )
#
#         x = self.x_update(
#             h=h,
#             coord=x,
#             edge_index=edge_index,
#             coord_diff=coord_diff,
#             edge_attr=edge_attr,
#             node_mask=node_mask,
#             edge_mask=edge_mask,
#         )
#
#         x = x.view(bs, n_nodes, 3)
#         h = h.view(bs, n_nodes, 8)
#
#         return x, h


class GCL(nn.Module):
    """Graph Convolution layer based on aggregation"""

    def __init__(
        self,
        input_nf: int,
        output_nf: int,
        hidden_nf: int,
        normalization_factor: float = 100.0,
        edges_in_d: int = 0,
        nodes_att_dim: int = 0,
    ):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, output_nf),
        )

        self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

        # Exact residual identity at init
        nn.init.zeros_(self.node_mlp[-1].weight)
        nn.init.zeros_(self.node_mlp[-1].bias)

    def edge_model(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        att_val = self.att_mlp(mij)
        out = mij * att_val

        out = out * edge_mask
        return out, mij

    def node_model(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row = edge_index[0]

        agg = unsorted_segment_sum(
            data=edge_attr,
            segment_ids=row,
            num_segments=x.size(0),
            normalization_factor=self.normalization_factor,
        )

        agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        row = edge_index[0]
        col = edge_index[1]

        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat)
        h = h * node_mask

        return h


class EquivariantUpdate(nn.Module):
    def __init__(
        self,
        hidden_nf: int,
        normalization_factor: float = 100.0,
        edges_in_d: int = 1,
        coords_range: float = 10.0,
    ):
        super(EquivariantUpdate, self).__init__()

        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            layer,
        )
        nn.init.zeros_(self.coord_mlp[-1].weight)
        self.normalization_factor = normalization_factor

    def coord_model(
        self,
        h: torch.Tensor,
        coord: torch.Tensor,
        edge_index: torch.Tensor,
        coord_diff: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        row = edge_index[0]
        col = edge_index[1]
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)

        trans = coord_diff * self.coord_mlp(input_tensor)

        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(
            data=trans,
            segment_ids=row,
            num_segments=coord.size(0),
            normalization_factor=self.normalization_factor,
        )
        coord = coord + agg
        return coord

    def forward(
        self,
        h: torch.Tensor,
        coord: torch.Tensor,
        edge_index: torch.Tensor,
        coord_diff: torch.Tensor,
        edge_attr: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        coord = coord * node_mask
        return coord


def unsorted_segment_sum(
    data: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    normalization_factor: float,
) -> torch.Tensor:
    """
    Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Normalization: 'sum'.
    """

    result = torch.zeros(
        (num_segments, data.size(1)), dtype=data.dtype, device=data.device
    )
    segment_ids = segment_ids.unsqueeze(-1).expand_as(data)

    result.scatter_add_(0, segment_ids, data)
    result = result / normalization_factor

    return result


def get_adj_matrix(n_nodes: int, batch_size: int, device: torch.device) -> torch.Tensor:
    # Generate batch offsets
    batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * n_nodes

    # Generate row and column indices for a single batch
    row_indices = torch.arange(n_nodes, device=device).repeat(n_nodes, 1).T.flatten()
    col_indices = torch.arange(n_nodes, device=device).repeat(n_nodes)

    # Expand to all batches
    rows = (row_indices.unsqueeze(0) + batch_offsets).flatten()
    cols = (col_indices.unsqueeze(0) + batch_offsets).flatten()

    # Store the edges as LongTensor
    edges = torch.stack(
        [
            rows.long(),
            cols.long(),
        ],
        dim=0,
    ).to(device)

    return edges


def coord2diff(
    x: torch.Tensor, edge_index: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    row = edge_index[0]
    col = edge_index[1]

    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / norm

    return radial, coord_diff
