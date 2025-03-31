import random

import onnx
import onnxruntime
import torch
from rdkit import Chem

from ml_conformer_generator.ml_conformer_generator.utils import (
    prepare_adj_mat_seer_input,
)

session = onnxruntime.InferenceSession("./adj_mat_seer_chembl_15_39.onnx")

input_shapes = [x.shape for x in session.get_inputs()]
input_names = [x.name for x in session.get_inputs()]

output_names = [x.name for x in session.get_outputs()]

device = "cpu"


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

np_inputs = []
for m in inputs:
    np_inputs.append(m.numpy())

elements, dist_mat, adj_mat = np_inputs


print(input_names)
print(input_shapes)
print(output_names)

out = session.run(
    None, {"elements": elements, "dist_mat": dist_mat, "adj_mat": adj_mat}
)

print(out)
