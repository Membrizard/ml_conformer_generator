from typing import Tuple

import torch
import torch.nn as nn


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

        return h, mij


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


class EquivariantBlock(nn.Module):
    def __init__(
        self,
        hidden_nf: int = 2,
        edge_feat_nf: int = 2,
        coords_range: float = 15.0,
        normalization_factor: float = 100.0,
    ):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.coords_range_layer = coords_range
        self.normalization_factor = normalization_factor

        self.gcl_0 = GCL(
            input_nf=hidden_nf,
            output_nf=hidden_nf,
            hidden_nf=hidden_nf,
            edges_in_d=edge_feat_nf,
            normalization_factor=normalization_factor,
        )

        self.gcl_1 = GCL(
            input_nf=hidden_nf,
            output_nf=hidden_nf,
            hidden_nf=hidden_nf,
            edges_in_d=edge_feat_nf,
            normalization_factor=normalization_factor,
        )

        self.gcl_equiv = EquivariantUpdate(
            hidden_nf=hidden_nf,
            edges_in_d=edge_feat_nf,
            coords_range=self.coords_range_layer,
            normalization_factor=normalization_factor,
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        distances, coord_diff = coord2diff(x, edge_index)

        edge_attr = torch.cat([distances, edge_attr], dim=1)

        h, _ = self.gcl_0(
            h=h,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        h, _ = self.gcl_1(
            h=h,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        x = self.gcl_equiv(
            h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask
        )

        h = h * node_mask
        return h, x


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf: int,
        hidden_nf: int,
        coords_range: float = 15.0,
        normalization_factor: float = 100.0,
    ):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.normalization_factor = normalization_factor

        edge_feat_nf = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, in_node_nf)

        self.e_block_0 = EquivariantBlock(
            hidden_nf=hidden_nf,
            edge_feat_nf=edge_feat_nf,
            coords_range=coords_range,
            normalization_factor=self.normalization_factor,
        )

        self.e_block_1 = EquivariantBlock(
            hidden_nf=hidden_nf,
            edge_feat_nf=edge_feat_nf,
            coords_range=coords_range,
            normalization_factor=self.normalization_factor,
        )

        self.e_block_2 = EquivariantBlock(
            hidden_nf=hidden_nf,
            edge_feat_nf=edge_feat_nf,
            coords_range=coords_range,
            normalization_factor=self.normalization_factor,
        )

        self.e_block_3 = EquivariantBlock(
            hidden_nf=hidden_nf,
            edge_feat_nf=edge_feat_nf,
            coords_range=coords_range,
            normalization_factor=self.normalization_factor,
        )

        self.e_block_4 = EquivariantBlock(
            hidden_nf=hidden_nf,
            edge_feat_nf=edge_feat_nf,
            coords_range=coords_range,
            normalization_factor=self.normalization_factor,
        )

        self.e_block_5 = EquivariantBlock(
            hidden_nf=hidden_nf,
            edge_feat_nf=edge_feat_nf,
            coords_range=coords_range,
            normalization_factor=self.normalization_factor,
        )

        self.e_block_6 = EquivariantBlock(
            hidden_nf=hidden_nf,
            edge_feat_nf=edge_feat_nf,
            coords_range=coords_range,
            normalization_factor=self.normalization_factor,
        )

        self.e_block_7 = EquivariantBlock(
            hidden_nf=hidden_nf,
            edge_feat_nf=edge_feat_nf,
            coords_range=coords_range,
            normalization_factor=self.normalization_factor,
        )

        self.e_block_8 = EquivariantBlock(
            hidden_nf=hidden_nf,
            edge_feat_nf=edge_feat_nf,
            coords_range=coords_range,
            normalization_factor=self.normalization_factor,
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        distances, _ = coord2diff(x, edge_index)

        h = self.embedding(h)

        h, x = self.e_block_0(
            h=h,
            x=x,
            edge_index=edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attr=distances,
        )

        h, x = self.e_block_1(
            h=h,
            x=x,
            edge_index=edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attr=distances,
        )

        h, x = self.e_block_2(
            h=h,
            x=x,
            edge_index=edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attr=distances,
        )

        h, x = self.e_block_3(
            h=h,
            x=x,
            edge_index=edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attr=distances,
        )

        h, x = self.e_block_4(
            h=h,
            x=x,
            edge_index=edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attr=distances,
        )

        h, x = self.e_block_5(
            h=h,
            x=x,
            edge_index=edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attr=distances,
        )

        h, x = self.e_block_6(
            h=h,
            x=x,
            edge_index=edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attr=distances,
        )

        h, x = self.e_block_7(
            h=h,
            x=x,
            edge_index=edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attr=distances,
        )

        h, x = self.e_block_8(
            h=h,
            x=x,
            edge_index=edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attr=distances,
        )

        h = self.embedding_out(h)
        h = h * node_mask

        return h, x


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


def remove_mean_with_mask(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    n = torch.sum(node_mask, 1, keepdim=True)

    mean = torch.sum(x, dim=1, keepdim=True) / n
    x = x - mean * node_mask
    return x


class EGNNDynamics(nn.Module):
    def __init__(
        self,
        in_node_nf: int,
        context_node_nf: int,  # -> Calculated from context properties
        n_dims: int = 3,
        hidden_nf: int = 420,  # -> 420 our default
        device: torch.device = torch.device("cpu"),
        normalization_factor: float = 100.0,
    ):
        super().__init__()

        # Should have 9 Equivarinat blocks n_layers = 9
        self.egnn = EGNN(
            in_node_nf=in_node_nf + context_node_nf,
            hidden_nf=hidden_nf,
            normalization_factor=normalization_factor,
        )
        self.in_node_nf = in_node_nf

        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims

    def forward(self, t, xh, node_mask, edge_mask, context):
        bs, n_nodes, _ = xh.size()

        edges = self.get_adj_matrix(n_nodes, bs, self.device)

        node_mask = node_mask.view(bs * n_nodes, 1)
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask
        x = xh[:, 0 : self.n_dims].clone()

        h = xh[:, self.n_dims :].clone()

        h_time = t.view(bs, 1).repeat(1, n_nodes)
        h_time = h_time.view(bs * n_nodes, 1)

        h = torch.cat([h, h_time], dim=1)

        # Context is added for conditional generation

        context = context.view(bs * n_nodes, self.context_node_nf)

        h = torch.cat([h, context], dim=1)

        h_final, x_final = self.egnn(
            h=h, x=x, edge_index=edges, node_mask=node_mask, edge_mask=edge_mask
        )

        vel = (
            x_final - x
        ) * node_mask  # This masking operation is redundant but just in case

        h_final = h_final[:, : -self.context_node_nf]

        h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        h_final = h_final.view(bs, n_nodes, -1)

        return torch.cat([vel, h_final], dim=2)

    @staticmethod
    def get_adj_matrix(
        n_nodes: int, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        # Generate batch offsets
        batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * n_nodes

        # Generate row and column indices for a single batch
        row_indices = (
            torch.arange(n_nodes, device=device).repeat(n_nodes, 1).T.flatten()
        )
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
