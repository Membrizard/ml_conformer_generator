from pathlib import Path

from typing import Tuple
from mlconfgen import MLConformerGenerator
from .onnx_export_utils import egnn_onnx_export, adj_mat_seer_onnx_export

MOCK_MOLECULES = ("./mol_examples/ceyyag.xyz", "./mol_examples/cpromz.xyz")


def export_to_onnx(
    model: MLConformerGenerator,
    egnn_save_path: str = "./egnn_chembl_15_39.onnx",
    adj_mat_seer_save_path: str = "./adj_mat_seer_chembl_15_39.onnx",
    mock_molecules: Tuple[str] = MOCK_MOLECULES,
) -> None:
    """
    Exports the model to ONNX format.
    When running export it is recommended to create a model on cpu MLConformerGenerator(device="cpu"),
    if using accelerators indicate the exact device i.e
    MLConformerGenerator(device="mps:0") or MLConformerGenerator(device="cuda:0")
    :param model: MLConformer generator instance with loaded weights
    :param egnn_save_path: save path for EGNN model in ONNX format
    :param adj_mat_seer_save_path: save path for AdjMatSeer model in ONNX format
    :param mock_molecules: a list of paths to mock molecules to use as dummy pass for AdjMatSeer conversion
    :return: Exports Denoising EGNN and AdjMatSeer to ONNX to make them compatible with ONNX runtime.
    To Load ONNX model use MLConformerGeneratorONNX a PyTorch - free ONNX based implementation.

    """

    base_path = Path(__file__).parent

    egnn_save_path = Path(egnn_save_path)
    adj_mat_seer_save_path = Path(adj_mat_seer_save_path)

    m = [str(base_path / x) for x in mock_molecules]

    egnn_onnx_export(generative_model=model.generative_model, save_path=egnn_save_path)
    adj_mat_seer_onnx_export(
        adj_mat_seer=model.adj_mat_seer,
        save_path=adj_mat_seer_save_path,
        mock_molecules=m,
    )
    return None
