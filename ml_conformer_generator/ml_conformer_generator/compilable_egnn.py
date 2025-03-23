import typing

import torch
import torch.nn as nn
import math

# import numpy as np

from typing import Tuple


class GCL(nn.Module):
    """Graph Convolution layer based on aggregation"""

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        normalization_factor,
        # aggregation_method,
        edges_in_d=0,
        nodes_att_dim=0,
        # act_fn=nn.SiLU(),
        # attention=False,  # True by default in the final model
    ):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        # self.aggregation_method = aggregation_method
        # self.attention = attention

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

        # if self.attention:
        self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        # if edge_attr is None:  # Unused.
        #     out = torch.cat([source, target], dim=1)
        # else:
        out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        # if self.attention:
        att_val = self.att_mlp(mij)
        out = mij * att_val
        # else:
        #     out = mij

        # if edge_mask is not None:
        out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr,
                   # node_attr
                   ):
        row = edge_index[0]
        # col = edge_index[1]

        agg = unsorted_segment_sum(
            edge_attr,
            row,
            num_segments=x.size(0),
            normalization_factor=self.normalization_factor,
            # aggregation_method=self.aggregation_method,
        )
        # if node_attr is not None:
        #     agg = torch.cat([x, agg, node_attr], dim=1)
        # else:
        agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(
        self,
        h,
        edge_index,
        edge_attr,
        # node_attr=None,
        node_mask,
        edge_mask,
    ):
        row = edge_index[0]
        col = edge_index[1]

        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat)
        # if node_mask is not None:
        h = h * node_mask
        return h, mij


class EquivariantUpdate(nn.Module):
    def __init__(
        self,
        hidden_nf,
        normalization_factor,
        # aggregation_method,
        edges_in_d=1,
        act_fn=nn.SiLU(),
        coords_range=10.0,
    ):
        super(EquivariantUpdate, self).__init__()

        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer,
        )
        self.normalization_factor = normalization_factor
        # self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask):
        row = edge_index[0]
        col = edge_index[1]
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)

        trans = coord_diff * self.coord_mlp(input_tensor)

        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(
            trans,
            row,
            num_segments=coord.size(0),
            normalization_factor=self.normalization_factor,
            # aggregation_method=self.aggregation_method,
        )
        coord = coord + agg
        return coord

    def forward(
        self,
        h,
        coord,
        edge_index,
        coord_diff,
        edge_attr,
        node_mask,
        edge_mask,
    ):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        # if node_mask is not None:
        coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(
        self,
        hidden_nf,
        edge_feat_nf=2,
        # device="cpu",
        # act_fn=nn.SiLU(),
        # n_layers=2,
        # attention=True,
        norm_diff=True,
        coords_range=15,
        norm_constant=1,
        # sin_embedding=None,
        normalization_factor=100,
        # aggregation_method="sum",
    ):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        # self.device = device
        # self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        # self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        # self.aggregation_method = aggregation_method

        self.gcl_0 = GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=edge_feat_nf,
                    # act_fn=act_fn,
                    # attention=attention,
                    normalization_factor=self.normalization_factor,
                    # aggregation_method=self.aggregation_method,
                )

        self.gcl_1 = GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=edge_feat_nf,
                    # act_fn=act_fn,
                    # attention=attention,
                    normalization_factor=self.normalization_factor,
                    # aggregation_method=self.aggregation_method,
                )

        self.gcl_equiv = EquivariantUpdate(
                hidden_nf,
                edges_in_d=edge_feat_nf,
                act_fn=nn.SiLU(),
                coords_range=self.coords_range_layer,
                normalization_factor=self.normalization_factor,
                # aggregation_method=self.aggregation_method,
            )

        # for i in range(0, n_layers):
        #     self.add_module(
        #         f"gcl_{i}",
        #         GCL(
        #             self.hidden_nf,
        #             self.hidden_nf,
        #             self.hidden_nf,
        #             edges_in_d=edge_feat_nf,
        #             # act_fn=act_fn,
        #             attention=attention,
        #             normalization_factor=self.normalization_factor,
        #             # aggregation_method=self.aggregation_method,
        #         ),
        #     )
        # self.add_module(
        #     "gcl_equiv",
        #     EquivariantUpdate(
        #         hidden_nf,
        #         edges_in_d=edge_feat_nf,
        #         act_fn=nn.SiLU(),
        #         coords_range=self.coords_range_layer,
        #         normalization_factor=self.normalization_factor,
        #         # aggregation_method=self.aggregation_method,
        #     ),
        # )
        # self.to(self.device)

    def forward(self, h, x, edge_index, node_mask, edge_mask, edge_attr):
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        # if self.sin_embedding is not None:
        #     distances = self.sin_embedding(distances)
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

        # for i in range(0, self.n_layers):
        #     h, _ = self._modules["gcl_%d" % i](
        #         h=h,
        #         edge_index=edge_index,
        #         edge_attr=edge_attr,
        #         node_mask=node_mask,
        #         edge_mask=edge_mask,
        #     )
        # x = self._modules["gcl_equiv"](
        #     h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask
        # )

        # Important, the bias of the last linear might be non-zero

        # if node_mask is not None:
        h = h * node_mask
        return h, x


# class SinEmbedding(nn.Module):
#     def __init__(self, max_res=15.0, min_res=15.0 / 2000.0, div_factor=4):
#         super().__init__()
#         self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
#         self.frequencies = (
#             2 * math.pi * div_factor ** torch.arange(self.n_frequencies) / max_res
#         )
#         self.dim = len(self.frequencies) * 2
#
#     def forward(self, x):
#         x = torch.sqrt(x + 1e-8)
#         emb = x * self.frequencies[None, :].to(x.device)
#         emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
#         return emb.detach()


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        hidden_nf,
        # device="cpu",
        # act_fn=nn.SiLU(),
        # n_layers=3,
        # attention=False,
        norm_diff=True,
        # out_node_nf=None,
        coords_range=15,
        norm_constant=1,
        # inv_sublayers=2,
        # sin_embedding=False,
        normalization_factor=100,
        # aggregation_method="sum",
    ):
        super(EGNN, self).__init__()
        # if out_node_nf is None:
        #     out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        # self.device = device
        # n_layers = 3
        # self.coords_range_layer = float(coords_range / n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        # self.aggregation_method = aggregation_method
        self.norm_constant = norm_constant

        # if sin_embedding:
        #     self.sin_embedding = SinEmbedding()
        #     edge_feat_nf = self.sin_embedding.dim * 2
        # else:
        #     self.sin_embedding = None
        edge_feat_nf = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, in_node_nf)

        self.e_block_0 = EquivariantBlock(
                    hidden_nf,
                    edge_feat_nf=edge_feat_nf,
                    # device=device,
                    # act_fn=act_fn,
                    # n_layers=inv_sublayers,
                    # attention=attention,
                    norm_diff=norm_diff,
                    coords_range=coords_range,
                    norm_constant=norm_constant,
                    # sin_embedding=self.sin_embedding,
                    normalization_factor=self.normalization_factor,
                    # aggregation_method=self.aggregation_method,
                )

        self.e_block_1 = EquivariantBlock(
            hidden_nf,
            edge_feat_nf=edge_feat_nf,
            # device=device,
            # act_fn=act_fn,
            # n_layers=inv_sublayers,
            # attention=attention,
            norm_diff=norm_diff,
            coords_range=coords_range,
            norm_constant=norm_constant,
            # sin_embedding=self.sin_embedding,
            normalization_factor=self.normalization_factor,
            # aggregation_method=self.aggregation_method,
        )

        self.e_block_2 = EquivariantBlock(
            hidden_nf,
            edge_feat_nf=edge_feat_nf,
            # device=device,
            # act_fn=act_fn,
            # n_layers=inv_sublayers,
            # attention=attention,
            norm_diff=norm_diff,
            coords_range=coords_range,
            norm_constant=norm_constant,
            # sin_embedding=self.sin_embedding,
            normalization_factor=self.normalization_factor,
            # aggregation_method=self.aggregation_method,
        )

        self.e_block_3 = EquivariantBlock(
            hidden_nf,
            edge_feat_nf=edge_feat_nf,
            # device=device,
            # act_fn=act_fn,
            # n_layers=inv_sublayers,
            # attention=attention,
            norm_diff=norm_diff,
            coords_range=coords_range,
            norm_constant=norm_constant,
            # sin_embedding=self.sin_embedding,
            normalization_factor=self.normalization_factor,
            # aggregation_method=self.aggregation_method,
        )

        self.e_block_4 = EquivariantBlock(
            hidden_nf,
            edge_feat_nf=edge_feat_nf,
            # device=device,
            # act_fn=act_fn,
            # n_layers=inv_sublayers,
            # attention=attention,
            norm_diff=norm_diff,
            coords_range=coords_range,
            norm_constant=norm_constant,
            # sin_embedding=self.sin_embedding,
            normalization_factor=self.normalization_factor,
            # aggregation_method=self.aggregation_method,
        )

        self.e_block_5 = EquivariantBlock(
            hidden_nf,
            edge_feat_nf=edge_feat_nf,
            # device=device,
            # act_fn=act_fn,
            # n_layers=inv_sublayers,
            # attention=attention,
            norm_diff=norm_diff,
            coords_range=coords_range,
            norm_constant=norm_constant,
            # sin_embedding=self.sin_embedding,
            normalization_factor=self.normalization_factor,
            # aggregation_method=self.aggregation_method,
        )

        self.e_block_6 = EquivariantBlock(
            hidden_nf,
            edge_feat_nf=edge_feat_nf,
            # device=device,
            # act_fn=act_fn,
            # n_layers=inv_sublayers,
            # attention=attention,
            norm_diff=norm_diff,
            coords_range=coords_range,
            norm_constant=norm_constant,
            # sin_embedding=self.sin_embedding,
            normalization_factor=self.normalization_factor,
            # aggregation_method=self.aggregation_method,
        )

        self.e_block_7 = EquivariantBlock(
            hidden_nf,
            edge_feat_nf=edge_feat_nf,
            # device=device,
            # act_fn=act_fn,
            # n_layers=inv_sublayers,
            # attention=attention,
            norm_diff=norm_diff,
            coords_range=coords_range,
            norm_constant=norm_constant,
            # sin_embedding=self.sin_embedding,
            normalization_factor=self.normalization_factor,
            # aggregation_method=self.aggregation_method,
        )

        self.e_block_8 = EquivariantBlock(
            hidden_nf,
            edge_feat_nf=edge_feat_nf,
            # device=device,
            # act_fn=act_fn,
            # n_layers=inv_sublayers,
            # attention=attention,
            norm_diff=norm_diff,
            coords_range=coords_range,
            norm_constant=norm_constant,
            # sin_embedding=self.sin_embedding,
            normalization_factor=self.normalization_factor,
            # aggregation_method=self.aggregation_method,
        )
        # for i in range(0, n_layers):
        #     self.add_module(
        #         "e_block_%d" % i,
        #         EquivariantBlock(
        #             hidden_nf,
        #             edge_feat_nf=2,
        #             device=device,
        #             act_fn=act_fn,
        #             n_layers=inv_sublayers,
        #             attention=attention,
        #             norm_diff=norm_diff,
        #             coords_range=coords_range,
        #             norm_constant=norm_constant,
        #             sin_embedding=self.sin_embedding,
        #             normalization_factor=self.normalization_factor,
        #             aggregation_method=self.aggregation_method,
        #         ),
        #     )
        # self.to(self.device)

    def forward(self, h, x, edge_index, node_mask, edge_mask) -> typing.Tuple[torch.Tensor, torch.Tensor]:

        distances, _ = coord2diff(x, edge_index, self.norm_constant)
        # if self.sin_embedding is not None:
        #     distances = self.sin_embedding(distances)
        h = self.embedding(h)

        h, x = self.e_block_0(
            h,
            x,
            edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attr=distances,
        )

        h, x = self.e_block_1(
            h,
            x,
            edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attr=distances,
        )

        h, x = self.e_block_2(
            h,
            x,
            edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attr=distances,
        )

        h, x = self.e_block_3(
            h,
            x,
            edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attr=distances,
        )

        h, x = self.e_block_4(
            h,
            x,
            edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attr=distances,
        )

        h, x = self.e_block_5(
            h,
            x,
            edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attr=distances,
        )

        h, x = self.e_block_6(
            h,
            x,
            edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attr=distances,
        )

        h, x = self.e_block_7(
            h,
            x,
            edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attr=distances,
        )

        h, x = self.e_block_8(
            h,
            x,
            edge_index,
            node_mask=node_mask,
            edge_mask=edge_mask,
            edge_attr=distances,
        )
        # for i in range(0, self.n_layers):
        #     h, x = self._modules["e_block_%d" % i](
        #         h,
        #         x,
        #         edge_index,
        #         node_mask=node_mask,
        #         edge_mask=edge_mask,
        #         edge_attr=distances,
        #     )

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        # if node_mask is not None:
        h = h * node_mask
        return h, x


def coord2diff(x: torch.Tensor, edge_index: torch.Tensor, norm_constant: int) -> Tuple[torch.Tensor, torch.Tensor]:
    row = edge_index[0]
    col = edge_index[1]

    coord_diff = x[row] - x[col]
    radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff / (norm + norm_constant)
    return radial, coord_diff

# OLD
# def unsorted_segment_sum(
#     data, segment_ids, num_segments, normalization_factor,
#         # aggregation_method: str
# ):
#     """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
#     Normalization: 'sum' or 'mean'.
#     """
#     result_shape = [num_segments, data.size(1)]
#     result = data.new_full(result_shape, 0)  # Init empty result tensor.
#     segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
#     result.scatter_add_(0, segment_ids, data)
#     result = result / normalization_factor
#
#     # if aggregation_method == "mean":
#     #     norm = data.new_zeros(result.shape)
#     #     norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
#     #     norm[norm == 0] = 1
#     #     result = result / norm
#     return result


# New
def unsorted_segment_sum(
        data: torch.Tensor,
        segment_ids: torch.Tensor,
        num_segments: int,
        normalization_factor: int
) -> torch.Tensor:
    """
    Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Normalization: 'sum'.
    """

    result = torch.zeros((num_segments, data.size(1)), dtype=data.dtype, device=data.device)
    segment_ids = segment_ids.unsqueeze(-1).expand_as(data)

    result.scatter_add_(0, segment_ids, data)
    result = result / normalization_factor

    return result


# def remove_mean(x):
#     mean = torch.mean(x, dim=1, keepdim=True)
#     x = x - mean
#     return x


def remove_mean_with_mask(x, node_mask):
    # Remove rdundant checks
    # masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    # assert masked_max_abs_value < 1e-5, f"Error {masked_max_abs_value} too high"
    # N = node_mask.sum(1, keepdims=True)
    N = torch.sum(node_mask, 1, keepdim=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


# --------------------------------------------------------------------------------
# Default EGNNDynamics parameters:
# EGNN_dynamics_QM9(
#         in_node_nf=dynamics_in_node_nf,
#         context_node_nf=args.context_node_nf,
#         n_dims=3,
#         device=device,
#         hidden_nf=args.nf, -> 192
#         act_fn=torch.nn.SiLU(),
#         n_layers=args.n_layers, -> 9
#         attention=args.attention, -> True
#         tanh=args.tanh, -> False
#         mode=args.model, -> egnn_dynamics removed
#         norm_constant=args.norm_constant, -> 0
#         inv_sublayers=args.inv_sublayers, -> 2
#         sin_embedding=args.sin_embedding, -> False
#         normalization_factor=args.normalization_factor, -> 100
#         aggregation_method=args.aggregation_method -> sum
#         )


class EGNNDynamics(nn.Module):
    def __init__(
        self,
        in_node_nf,
        context_node_nf,  # -> Calculated from context properties
        n_dims: int = 3,
        hidden_nf: int = 420,  # -> 420 our default
        device: torch.device = torch.device("cpu"),
        # act_fn=torch.nn.SiLU(),
        # n_layers: int = 9,
        # attention=True,
        # condition_time=True,
        norm_constant=0,
        # inv_sublayers=2,
        # sin_embedding=False,
        normalization_factor=100,
        # aggregation_method="sum",
    ):
        super().__init__()

        # Should have 9 Equivarinat blocks n_layers = 9
        self.egnn = EGNN(
            in_node_nf=in_node_nf + context_node_nf,
            hidden_nf=hidden_nf,
            # device=device,
            # act_fn=act_fn,
            # n_layers=n_layers,
            # attention=attention,
            norm_constant=norm_constant,
            # inv_sublayers=inv_sublayers,
            # sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            # aggregation_method=aggregation_method,
        )
        self.in_node_nf = in_node_nf

        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims

        # def create_edges_dict() -> typing.Dict[int, typing.List[torch.LongTensor]]:
        #     return {}

        # self._edges_dict = {0: {0: torch.tensor(0)}}

        # self.condition_time = condition_time

    # def forward(self, t, xh, node_mask, edge_mask, context):
    #     raise NotImplementedError
    #
    # def wrap_forward(self, node_mask, edge_mask, context):
    #     def fwd(time, state):
    #         return self._forward(time, state, node_mask, edge_mask, context)
    #
    #     return fwd
    #
    # def unwrap_forward(self):
    #     return self._forward

    def forward(self, t, xh, node_mask, edge_mask, context):
        # print(f"t - size {t.size()} type {t.type()}")
        # print(t)
        # print(f"xh - size {xh.size()} type {xh.type()}")
        # print(xh)
        # print(f"node_mask - size {node_mask.size()} type {node_mask.type()}")
        # print(node_mask)
        # print(f"edge_mask - size {edge_mask.size()} type {edge_mask.type()}")
        # print(edge_mask)
        # print(f"context - size {context.size()} type {context.type()}")
        # print(context)


        # bs, n_nodes, dims = xh.shape
        # bs, n_nodes, dims = xh.size()
        bs, n_nodes, _ = xh.size()


        # h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)

        # edges = torch.zeros((20, 20))
        # edges = [x.to(self.device) for x in edges]
        node_mask = node_mask.view(bs * n_nodes, 1)
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask
        x = xh[:, 0: self.n_dims].clone()
        # if h_dims == 0:
        #     h = torch.ones(bs * n_nodes, 1).to(self.device)
        # else:
        h = xh[:, self.n_dims:].clone()

        # Condition time is set to True by default
        # if self.condition_time:
        # if np.prod(t.size()) == 1:

        # A Case of single sample generation will deprecate - Add Notes or cover in higher level
        # if t.numel() == 1:
        #     print("A")
        #     # Case of a single sample generation
        #     # t is the same for all elements in batch.
        #     h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
        # else:
        #     print("B")
        # t is different over the batch dimension by default for multiple sample eneration
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

        # if context is not None:
        # Slice off context size:
        h_final = h_final[:, : -self.context_node_nf]

        # if self.condition_time:
        # Slice off last dimension which represented time.
        h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        # Deprecate redundant check for speed up
        # if torch.any(torch.isnan(vel)):
        #     print("Warning: detected nan, resetting EGNN output to zero.")
        #     vel = torch.zeros_like(vel)

        # if node_mask is None:
        #     vel = remove_mean(vel)
        # else:
        vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        # Not reacheable during generation
        # if h_dims == 0:
        #     print("C")
        #     return vel
        # else:

        h_final = h_final.view(bs, n_nodes, -1)
        return torch.cat([vel, h_final], dim=2)

    # OLD
    # def get_adj_matrix(self, n_nodes: int, batch_size: int, device):
    #     if n_nodes in self._edges_dict:
    #         edges_dic_b = self._edges_dict[n_nodes]
    #         if batch_size in edges_dic_b:
    #             return edges_dic_b[batch_size]
    #         else:
    #             # get edges for a single sample
    #             # rows, cols = [], []
    #             # for batch_idx in range(batch_size):
    #             #     for i in range(n_nodes):
    #             #         for j in range(n_nodes):
    #             #             rows.append(i + batch_idx * n_nodes)
    #             #             cols.append(j + batch_idx * n_nodes)
    #             # Create a tensor of batch indices
    #             batch_offsets = torch.arange(batch_size).unsqueeze(1) * n_nodes
    #
    #             # Generate row and column indices for a single batch
    #             row_indices = torch.arange(n_nodes).repeat(n_nodes, 1).T.flatten()
    #             col_indices = torch.arange(n_nodes).repeat(n_nodes)
    #
    #             # Expand to all batches
    #             rows = (row_indices.unsqueeze(0) + batch_offsets).flatten()
    #             cols = (col_indices.unsqueeze(0) + batch_offsets).flatten()
    #
    #             edges = [
    #                 torch.LongTensor(rows).to(device),
    #                 torch.LongTensor(cols).to(device),
    #             ]
    #             edges_dic_b[batch_size] = edges
    #             return edges
    #     else:
    #         self._edges_dict[n_nodes] = {0: [torch.tensor([0])]}
    #         return self.get_adj_matrix(n_nodes, batch_size, device)

    # New with Cashing
    # def get_adj_matrix(self, n_nodes: int, batch_size: int, device: torch.device) -> torch.Tensor:
    #     # Check if the node number dictionary exists
    #     if n_nodes not in self._edges_dict:
    #         self._edges_dict[n_nodes] = {0: torch.tensor(0)}
    #     # Check if the batch size dictionary exists
    #     edges_dic_b = self._edges_dict[n_nodes]
    #     if batch_size in edges_dic_b:
    #         return edges_dic_b[batch_size]
    #
    #     # Generate batch offsets
    #     batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * n_nodes
    #
    #     # Generate row and column indices for a single batch
    #     row_indices = torch.arange(n_nodes, device=device).repeat(n_nodes, 1).T.flatten()
    #     col_indices = torch.arange(n_nodes, device=device).repeat(n_nodes)
    #
    #     # Expand to all batches
    #     rows = (row_indices.unsqueeze(0) + batch_offsets).flatten()
    #     cols = (col_indices.unsqueeze(0) + batch_offsets).flatten()
    #     # print(torch.LongTensor(rows).unsqueeze(0))
    #
    #     # Store the edges as LongTensor
    #
    #     # edges = torch.cat([
    #     #     rows.long().unsqueeze(0),
    #     #     cols.long().unsqueeze(0),
    #     # ], dim=0).to(device)
    #
    #     edges = torch.stack([
    #         rows.long(),
    #         cols.long(),
    #     ], dim=0).to(device)
    #
    #     edges_dic_b[batch_size] = edges
    #
    #     return edges

    # New without caching
    def get_adj_matrix(self, n_nodes: int, batch_size: int, device: torch.device) -> torch.Tensor:
        # Generate batch offsets
        batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * n_nodes

        # Generate row and column indices for a single batch
        row_indices = torch.arange(n_nodes, device=device).repeat(n_nodes, 1).T.flatten()
        col_indices = torch.arange(n_nodes, device=device).repeat(n_nodes)

        # Expand to all batches
        rows = (row_indices.unsqueeze(0) + batch_offsets).flatten()
        cols = (col_indices.unsqueeze(0) + batch_offsets).flatten()

        # Store the edges as LongTensor
        edges = torch.stack([
            rows.long(),
            cols.long(),
        ], dim=0).to(device)

        return edges
