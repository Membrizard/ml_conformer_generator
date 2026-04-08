import torch
from src.mlconfgen.adj_mat_seer import AdjMatSeer
from src.mlconfgen.rl_fine_tune.edm_adapter import EDMAdapter
from src.mlconfgen.rl_fine_tune.shared_prior_agent import SharedPriorAgent


# --- EDMAdaptor ---
def test_equivariant_block_forward_shapes():
    torch.manual_seed(42)
    hidden_nf = 8
    N = 6
    adapter = EDMAdapter()
    h = torch.randn(1, N, hidden_nf)
    x = torch.randn(1, N, 3)
    x = x - x.mean(dim=1)
    node_mask = torch.ones(N, 1)
    edge_mask = torch.ones(N * N, 1)

    x_out, h_out, log_prob, aux = adapter(x, h, edge_mask, node_mask)
    assert h_out.shape == h.shape
    assert x_out.shape == x.shape

    # Equivariance Check
    centers = x_out.mean(dim=1)
    diff = torch.abs(torch.zeros_like(centers) - centers).sum()
    assert diff <= 1e-4


# --- AdjMatSeer ---


def test_bond_trainer_forward_shapes():
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

    agent = SharedPriorAgent(pretrained_model=model)
    elements = torch.randint(0, 35, (B, dim))
    dist_mat = torch.randn(B, dim, dim).abs() + torch.eye(dim)
    adj_mat = torch.eye(dim).unsqueeze(0).repeat(B, 1, 1)
    with torch.no_grad():
        a_out = agent.agent_forward(elements, dist_mat, adj_mat)
        p_out = agent.prior_forward(elements, dist_mat, adj_mat)
        assert a_out is not None
        assert p_out is not None
        assert a_out.shape == (B, dim, dim, num_bond_types)
        assert p_out.shape == (B, dim, dim, num_bond_types)
