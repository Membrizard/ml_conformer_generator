import random
from pathlib import Path
from typing import List

import torch
from torch.export import Dim
from rdkit import Chem

from mlconfgen.utils.config import CONTEXT_NORMS
from mlconfgen.utils.mol_utils import prepare_adj_mat_seer_input


def egnn_onnx_export(
    generative_model: torch.nn.Module,
    save_path: Path,
    dummy_ref_context: torch.Tensor = torch.tensor(
        [53.6424, 108.3042, 151.4399], dtype=torch.float32
    ),
    context_norms: dict = CONTEXT_NORMS,
) -> None:
    egnn = generative_model.dynamics

    context_norms = {key: torch.tensor(value) for key, value in context_norms.items()}

    egnn_inputs = prepare_egnn_dummy_input(
        generative_model, dummy_ref_context, context_norms
    )

    batch_size = Dim("batch_size")
    num_nodes = Dim("num_nodes")
    num_edges = Dim("num_edges")

    try:
        export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
        onnx_model = torch.onnx.export(
            egnn,
            egnn_inputs,
            input_names=["t", "xh", "node_mask", "edge_mask", "context"],
            output_names=["out"],
            export_options=export_options,
            export_params=True,
            dynamic_shapes={
                "t": {0: batch_size},
                "xh": {0: batch_size, 1: num_nodes},
                "node_mask": {0: batch_size, 1: num_nodes},
                "edge_mask": {0: num_edges},
                "context": {0: batch_size, 1: num_nodes},
                "out": {0: batch_size, 1: num_nodes},
            },
            opset_version=18,
            verbose=True,
            dynamo=True,
        )
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            'Failed to export ONNX model. To resolve run `pip install "mlconfgen[onnx]"`\n'
        ) from e

    onnx_model.save(save_path)

    return None


def prepare_egnn_dummy_input(
    generative_model: torch.nn.Module,
    reference_context: torch.Tensor,
    context_norms: dict,
    n_samples: int = 2,
    min_n_nodes: int = 16,
    max_n_nodes: int = 20,
    s: int = 50,
    timesteps: int = 100,
):
    device = generative_model.dynamics.device
    nodesxsample = []

    for n in range(n_samples):
        nodesxsample.append(random.randint(min_n_nodes, max_n_nodes))

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

    z = generative_model.sample_combined_position_feature_noise(
        n_samples, max_n_nodes, node_mask
    )
    s_array = torch.full([n_samples, 1], fill_value=s, device=device)
    t_array = s_array + 1.0
    t_array = t_array / timesteps

    return t_array, z, node_mask, edge_mask, batch_context


def adj_mat_seer_onnx_export(
    adj_mat_seer: torch.nn.Module,
    save_path: Path,
    mock_molecules: List[str],
) -> None:
    mols = [Chem.MolFromXYZFile(x) for x in mock_molecules]

    device = adj_mat_seer.device
    dimension = adj_mat_seer.dimension

    inputs = prepare_adj_mat_seer_input(
        mols=mols,
        dimension=dimension,
        device=device,
    )

    batch_size = Dim("batch_size")

    export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
    onnx_model = torch.onnx.export(
        adj_mat_seer,
        inputs[:-1],
        input_names=["elements", "dist_mat", "adj_mat"],
        output_names=["out"],
        export_options=export_options,
        export_params=True,
        dynamic_shapes={
            "elements": {0: batch_size},
            "dist_mat": {0: batch_size},
            "adj_mat": {0: batch_size},
            "out": {0: batch_size},
        },
        opset_version=18,
        verbose=True,
        dynamo=True,
    )

    onnx_model.save(save_path)

    return None
