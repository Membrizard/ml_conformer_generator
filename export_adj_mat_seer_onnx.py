import random

import torch
from rdkit import Chem
from torch.export import Dim

from ml_conformer_generator.ml_conformer_generator.adj_mat_seer import AdjMatSeer
from ml_conformer_generator.ml_conformer_generator.utils import (
    prepare_adj_mat_seer_input,
)

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


def prepare_adj_mat_seer_dummy_input(device):
    mock_samples = ["./ceyyag.xyz", "cpromz.xyz", "yibfeu.xyz"]

    mols = [Chem.MolFromXYZFile(x) for x in mock_samples]

    (
        el_batch,
        dm_batch,
        b_adj_mat_batch,
        canonicalised_samples,
    ) = prepare_adj_mat_seer_input(
        mols=mols,
        n_samples=len(mols),
        dimension=42,
        device=device,
    )
    return el_batch, dm_batch, b_adj_mat_batch


inputs = prepare_adj_mat_seer_dummy_input(device)


batch_size = Dim("batch_size")

export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
onnx_model = torch.onnx.export(
    adj_mat_seer,
    inputs,
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

onnx_model.save("adj_mat_seer_chembl_15_39.onnx")
