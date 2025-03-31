import torch
import torch.nn as nn

from .utils import DIMENSION, NUM_BOND_TYPES

# Dimension of Embedding Must be a number divisible by 8
EMBEDDING_DIM = 64
# Number of tokens to embed maximal atomic number + 1
NUM_EMBEDDINGS = 36


class GraphConv(nn.Module):
    """
    The graph convolutional operator inspired by `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper;
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dimension: int = 42,
        device: torch.device = "cpu",
    ):
        super(GraphConv, self).__init__()

        self.dimension = dimension
        self.linear = nn.Linear(in_features, out_features)
        self.device = device

    def l_norm(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        degree = adjacency_matrix.sum(dim=-1)
        inv_sqrt_degree = torch.rsqrt(degree.clamp(min=1e-12))
        l_norm = (
            inv_sqrt_degree.unsqueeze(-1)
            * adjacency_matrix
            * inv_sqrt_degree.unsqueeze(-2)
        )

        return l_norm.to(self.device)

    def forward(
        self,
        x: torch.Tensor,
        l_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: element-shielding embedding of size (batch_size, DIMENSION, EMBEDDING_DIM)
        :param l_norm: pre-computed l-norm for a graph batch using self.l_norm()
        :return Nodes embedding
        """
        x = self.linear(x)

        output = torch.bmm(l_norm, x)

        return output


class AdjMatSeer(nn.Module):
    """
    Suggests probabilities of atoms adjacency based on their type and coordinates
    """

    def __init__(
        self,
        dimension: int = DIMENSION,
        n_hidden: int = 2048,
        embedding_dim: int = EMBEDDING_DIM,
        num_embeddings: int = NUM_EMBEDDINGS,
        num_bond_types: int = NUM_BOND_TYPES,
        device: torch.device = "cpu",
    ):
        super().__init__()
        self.dimension = dimension
        self.embedding_dim = embedding_dim
        self.num_bond_types = num_bond_types
        self.device = device

        self.act = nn.ReLU()
        self.gcn1 = GraphConv(embedding_dim, n_hidden, device=device)
        self.gcn2 = GraphConv(n_hidden, n_hidden, device=device)
        self.gcn3 = GraphConv(n_hidden, n_hidden, device=device)
        self.gcn4 = GraphConv(n_hidden, n_hidden, device=device)

        self.resize = nn.Linear(n_hidden, dimension * num_bond_types)

        self.nodes_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.nodes_coord_fc = nn.Linear(dimension, dimension * embedding_dim)

        # ___________________________________________________

        self.gcn1_dm = GraphConv(embedding_dim, n_hidden, device=device)
        self.gcn2_dm = GraphConv(n_hidden, n_hidden, device=device)
        self.gcn3_dm = GraphConv(n_hidden, n_hidden, device=device)

        self.dm_resize = nn.Linear(n_hidden, 1)

        self.dm_nodes_embedding = torch.nn.Embedding(num_embeddings, embedding_dim)

    def forward(
        self,
        elements: torch.Tensor,
        dist_mat: torch.Tensor,
        adj_mat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass
        :return: Adjacency matrix batch (batch_size, DIMENSION, DIMENSION, NUM_BOND_TYPES)
        """

        dm_nodes_embedded = self.dm_nodes_embedding(elements)

        dm_l_norm = self.gcn1_dm.l_norm(adjacency_matrix=dist_mat)

        conv1_dm = self.act(self.gcn1_dm(x=dm_nodes_embedded, l_norm=dm_l_norm))
        conv2_dm = self.act(self.gcn2_dm(x=conv1_dm, l_norm=dm_l_norm))
        conv3_dm = self.act(
            self.gcn3_dm(x=conv2_dm, l_norm=dm_l_norm)
        )  # size (batch_size, DIMENSION, n_hidden)

        emb = self.dm_resize(conv3_dm).squeeze(-1)  # Bottleneck

        # _______________________________________________________________

        # Create nodes embedding
        nodes_embedded = self.nodes_embedding(
            elements
        )  # size (batch_size, DIMENSION, EMBEDDING_DIM)

        # Scale embedding
        nodes_weighted_emb = torch.reshape(
            self.nodes_coord_fc(emb),
            (nodes_embedded.size(dim=0), self.dimension, self.embedding_dim),
        )  # size (batch_size, DIMENSION, EMBEDDING_DIM)

        # Include Shielding Constants by elementwise addition of weighted shielding
        nodes_merged = (
            nodes_embedded + nodes_weighted_emb
        )  # size (batch_size, DIMENSION, EMBEDDING_DIM)

        l_norm = self.gcn1.l_norm(adjacency_matrix=adj_mat)

        conv1 = self.act(self.gcn1(x=nodes_merged, l_norm=l_norm))
        conv2 = self.act(self.gcn2(x=conv1, l_norm=l_norm))
        conv3 = self.act(self.gcn3(x=conv2, l_norm=l_norm))
        conv4 = self.act(
            self.gcn4(x=conv3, l_norm=l_norm)
        )  # size (batch_size, DIMENSION, n_hidden)

        scaled_res = self.resize(conv4)

        adjacency_matrix = torch.reshape(
            scaled_res,
            (scaled_res.shape[0], self.dimension, self.dimension, self.num_bond_types),
        )
        # Symmetrize the output
        adjacency_matrix = torch.add(
            torch.transpose(adjacency_matrix, 1, 2), adjacency_matrix
        )  # size of (batch_size, DIMENSION, DIMENSION, NUM_BOND_TYPES)

        return adjacency_matrix
