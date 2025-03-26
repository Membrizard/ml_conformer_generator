import random

import torch

from ml_conformer_generator.ml_conformer_generator.egnn import EGNNDynamics
from ml_conformer_generator.ml_conformer_generator.equivariant_diffusion import (
    EquivariantDiffusion, PredefinedNoiseSchedule
)

device = "cpu"
net_dynamics = EGNNDynamics(
    in_node_nf=9,
    context_node_nf=3,
    hidden_nf=420,
    device=device,
)

generative_model = EquivariantDiffusion(
    dynamics=net_dynamics,
    in_node_nf=8,
    timesteps=1000,
    noise_precision=1e-5,
)

generative_model.load_state_dict(
    torch.load(
        "./ml_conformer_generator/ml_conformer_generator/weights/edm_moi_chembl_15_39.weights",
        map_location=device,
    )
)

diffusion_steps = 100

# Update denoising steps for the Equivarinat Diffusion
generative_model.gamma = PredefinedNoiseSchedule(
            timesteps=diffusion_steps, precision=1e-5
        )

generative_model.timesteps = torch.flip(
            torch.arange(0, diffusion_steps, device=device), dims=[0]
        )

generative_model.T = diffusion_steps

generative_model.to(device)
generative_model.eval()


def prepare_dummy_input(device):
    reference_context = torch.tensor(
        [53.6424, 108.3042, 151.4399], dtype=torch.float32, device=device
    )
    context_norms = {
        "mean": torch.tensor(
            [105.0766, 473.1938, 537.4675], dtype=torch.float32, device=device
        ),
        "mad": torch.tensor(
            [52.0409, 219.7475, 232.9718], dtype=torch.float32, device=device
        ),
    }

    n_samples = 20
    min_n_nodes = 15
    max_n_nodes = 39

    # Create a random list of sizes between min_n_nodes and max_n_nodes of length n_samples
    nodesxsample = []

    for n in range(n_samples):
        nodesxsample.append(random.randint(min_n_nodes, max_n_nodes))

    nodesxsample = torch.tensor(nodesxsample)

    batch_size = nodesxsample.size(0)

    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0: nodesxsample[i]] = 1

    # Compute edge_mask

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(
        batch_size * max_n_nodes * max_n_nodes, 1
    ).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    normed_context = (
        (reference_context - context_norms["mean"]) / context_norms["mad"]
    ).to(device)

    batch_context = normed_context.unsqueeze(0).repeat(batch_size, 1)

    batch_context = (
        batch_context.unsqueeze(1).repeat(1, max_n_nodes, 1) * node_mask
    )
    return node_mask, edge_mask, batch_context


node_mask, edge_mask, context = prepare_dummy_input(device)


# export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
onnx_model = torch.onnx.export(
    generative_model,
    (node_mask, edge_mask, context),
    input_names=["node_mask", "edge_mask", "context"],
    output_names=["x", "h"],
    export_params=True,
    opset_version=18,
    verbose=True,
    dynamo=True,
)

onnx_model.optimize()
onnx_model.save("100_steps_edm_moi_chembl_15_39.onnx")
