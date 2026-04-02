import copy

import torch
import torch.nn as nn

from ..adj_mat_seer import AdjMatSeer


class SharedPriorAgent(nn.Module):
    """
    Module for Reinforcement Learning Fine-Tuning of AdjMatSeer head.
    """

    def __init__(self, pretrained_model: AdjMatSeer):
        super().__init__()

        self.dimension = pretrained_model.dimension
        self.embedding_dim = pretrained_model.embedding_dim
        self.num_bond_types = pretrained_model.num_bond_types
        self.device = pretrained_model.device
        self.act = pretrained_model.act

        self.gcn1 = pretrained_model.gcn1
        self.gcn2 = pretrained_model.gcn2
        self.gcn3 = pretrained_model.gcn3
        self.gcn4 = pretrained_model.gcn4

        self.nodes_embedding = pretrained_model.nodes_embedding
        self.nodes_coord_fc = pretrained_model.nodes_coord_fc

        self.gcn1_dm = pretrained_model.gcn1_dm
        self.gcn2_dm = pretrained_model.gcn2_dm
        self.gcn3_dm = pretrained_model.gcn3_dm
        self.dm_resize = pretrained_model.dm_resize
        self.dm_nodes_embedding = pretrained_model.dm_nodes_embedding

        self.prior_resize = pretrained_model.resize
        self.agent_resize = copy.deepcopy(pretrained_model.resize)

        for p in self.parameters():
            p.requires_grad_(False)

        for p in self.agent_resize.parameters():
            p.requires_grad_(True)

    def _compute_hidden(
        self,
        elements: torch.Tensor,
        dist_mat: torch.Tensor,
        adj_mat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dm_nodes_embedded = self.dm_nodes_embedding(elements)
        dm_l_norm = self.gcn1_dm.l_norm(adjacency_matrix=dist_mat)

        conv1_dm = self.act(self.gcn1_dm(x=dm_nodes_embedded, l_norm=dm_l_norm))
        conv2_dm = self.act(self.gcn2_dm(x=conv1_dm, l_norm=dm_l_norm))
        conv3_dm = self.act(self.gcn3_dm(x=conv2_dm, l_norm=dm_l_norm))

        emb = self.dm_resize(conv3_dm).squeeze(-1)

        nodes_embedded = self.nodes_embedding(elements)
        nodes_weighted_emb = torch.reshape(
            self.nodes_coord_fc(emb),
            (nodes_embedded.size(0), self.dimension, self.embedding_dim),
        )
        nodes_merged = nodes_embedded + nodes_weighted_emb

        l_norm = self.gcn1.l_norm(adjacency_matrix=adj_mat)

        conv1 = self.act(self.gcn1(x=nodes_merged, l_norm=l_norm))
        conv2 = self.act(self.gcn2(x=conv1, l_norm=l_norm))
        conv3 = self.act(self.gcn3(x=conv2, l_norm=l_norm))
        conv4 = self.act(self.gcn4(x=conv3, l_norm=l_norm))
        return conv4

    def convert_to_adj_mat(self, scaled_res: torch.Tensor) -> torch.Tensor:
        adjacency_matrix = torch.reshape(
            scaled_res,
            (scaled_res.shape[0], self.dimension, self.dimension, self.num_bond_types),
        )
        adjacency_matrix = adjacency_matrix + adjacency_matrix.transpose(1, 2)
        return adjacency_matrix

    @torch.no_grad()
    def prior_forward(
        self,
        elements: torch.Tensor,
        dist_mat: torch.Tensor,
        adj_mat: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._compute_hidden(elements, dist_mat, adj_mat)
        scaled_res = self.prior_resize(hidden)
        return self.convert_to_adj_mat(scaled_res)

    def agent_forward(
        self,
        elements: torch.Tensor,
        dist_mat: torch.Tensor,
        adj_mat: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._compute_hidden(elements, dist_mat, adj_mat)
        scaled_res = self.agent_resize(hidden)
        return self.convert_to_adj_mat(scaled_res)
