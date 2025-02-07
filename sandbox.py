import torch
from ml_conformer_generator import EGNNDynamics, EquivariantDiffusion

device = "cpu"


def generate_sample(
    device,
    generative_model,
    reference_context,
    context_norms: dict = CONTEXT_NORMS,
    n_samples=100,
    max_n_nodes=39,
    min_n_nodes=25,
    fix_noise=False,
):
    # Create a random list of sizes between min_n_nodes and max_n_nodes of length n_samples
    nodesxsample = []

    for n in range(n_samples):
        nodesxsample.append(random.randint(min_n_nodes, max_n_nodes))

    print(f"Generating {len(nodesxsample)} Samples")

    nodesxsample = torch.tensor(nodesxsample)

    batch_size = nodesxsample.size(0)

    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0 : nodesxsample[i]] = 1

    # Compute edge_mask

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    normed_context = (
        (reference_context - context_norms["mean"]) / context_norms["mad"]
    ).to(device)

    batch_context = normed_context.unsqueeze(0).repeat(batch_size, 1)

    batch_context = batch_context.unsqueeze(1).repeat(1, max_n_nodes, 1) * node_mask

    x, h = generative_model.sample(
        batch_size,
        max_n_nodes,
        node_mask,
        edge_mask,
        batch_context,
        fix_noise=fix_noise,
    )

    mols = samples_to_rdkit_mol(x, h, node_mask)

    return mols


net_dynamics = EGNNDynamics(
    in_node_nf=9,  # -> Number of possible atom types + 1 - C, N, O, F, P, S, Cl, Br + 1
    context_node_nf=3,  # -> 3 diagonal components of the principal Moment of Inertia Tensor
    hidden_nf=20,  # -> default 420
    device=device,
)

generative_model = EquivariantDiffusion(
    dynamics=net_dynamics,
    in_node_nf=8,  # -> Number of possible atom types - C, N, O, F, P, S, Cl, Br
    timesteps=4,  # -> default number of timesteps is 1000
    noise_precision=1e-5,
)

generative_model.to(device)

generative_model.load_state_dict(
    torch.load(
        "checkpoint/EDM_MODEL_1001.weights",
        map_location=device,
    )
)

generative_model.eval()

samples = generate_sample(
    device,
    generative_model,
    ref_context,
    context_norms=CONTEXT_NORMS,
    n_samples=2,
    min_n_nodes=ref_n_atoms - 2,
    max_n_nodes=ref_n_atoms + 2,
    fix_noise=False,
)
