import random

import torch

from ml_conformer_generator.ml_conformer_generator.adj_mat_seer import AdjMatSeer

device = "cpu"
adj_mat_seer = AdjMatSeer(
            dimension=42,
            n_hidden=2048,
            embedding_dim=64,
            num_embeddings=36,
            num_bond_types=5,
            device=device,
        )

adj_mat_seer.load_state_dict(
            torch.load(
                "./ml_conformer_generator/ml_conformer_generator/weights/adj_mat_seer_chembl_15_39.weights",
                map_location=device,
            )
        )

adj_mat_seer.to(device)
adj_mat_seer.eval()


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

    n_samples = 2
    min_n_nodes = 18
    max_n_nodes = 20

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
    return batch_size, max_n_nodes, node_mask, edge_mask, batch_context


n_samples, n_nodes, node_mask, edge_mask, context = prepare_dummy_input(device)



export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
onnx_model = torch.onnx.export(
    adj_mat_seer,
    (n_samples, n_nodes, node_mask, edge_mask, context),
    "adj_mat_seer_chembl_15_39.onnx",
    input_names=["n_samples", "n_nodes", "node_mask", "edge_mask", "context"],
    output_names=["x", "h"],
    export_options=export_options,
    opset_version=18,
    verbose=True,
    dynamo=True,
)

onnx_model.optimize()
onnx_model.save("opt_adj_mat_seer_chembl_15_39.onnx")
