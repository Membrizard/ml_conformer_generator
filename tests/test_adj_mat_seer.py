import torch

from src.mlconfgen.adj_mat_seer import AdjMatSeer, GraphConv

# --- GraphConv ---


def test_graph_conv_forward_shape():
    torch.manual_seed(42)
    B, dim = 2, 10
    in_f, out_f = 16, 32
    gc = GraphConv(in_features=in_f, out_features=out_f, dimension=dim)
    x = torch.randn(B, dim, in_f)
    adj = torch.eye(dim).unsqueeze(0).repeat(B, 1, 1)
    l_norm = gc.l_norm(adj)
    out = gc(x, l_norm)
    assert out.shape == (B, dim, out_f)


def test_graph_conv_l_norm_shape():
    torch.manual_seed(42)
    B, dim = 2, 10
    gc = GraphConv(in_features=16, out_features=32, dimension=dim)
    adj = torch.eye(dim).unsqueeze(0).repeat(B, 1, 1)
    l_norm = gc.l_norm(adj)
    assert l_norm.shape == (B, dim, dim)


# --- AdjMatSeer ---


def test_adj_mat_seer_forward_shape():
    torch.manual_seed(42)
    B, dim = 2, 10
    n_hidden, emb_dim = 32, 16
    num_bond_types = 5
    model = AdjMatSeer(
        dimension=dim,
        n_hidden=n_hidden,
        embedding_dim=emb_dim,
        num_bond_types=num_bond_types,
    )
    elements = torch.randint(0, 35, (B, dim))
    dist_mat = torch.randn(B, dim, dim).abs() + torch.eye(dim)
    adj_mat = torch.eye(dim).unsqueeze(0).repeat(B, 1, 1)
    out = model(elements, dist_mat, adj_mat)
    assert out.shape == (B, dim, dim, num_bond_types)


def test_adj_mat_seer_no_grad():
    torch.manual_seed(42)
    B, dim = 1, 10
    model = AdjMatSeer(
        dimension=dim,
        n_hidden=32,
        embedding_dim=16,
    )
    elements = torch.randint(0, 35, (B, dim))
    dist_mat = torch.randn(B, dim, dim).abs() + torch.eye(dim)
    adj_mat = torch.eye(dim).unsqueeze(0).repeat(B, 1, 1)
    with torch.no_grad():
        out = model(elements, dist_mat, adj_mat)
    assert out is not None
