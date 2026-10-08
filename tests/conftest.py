from pathlib import Path

import numpy as np
import pytest
import torch
from rdkit import Chem

from src.mlconfgen.utils.config import ATOM_DECODER, CONTEXT_NORMS

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def seed_rng():
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np.random.seed(42)
    torch.manual_seed(42)
    try:
        yield 42
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)



def _weights_present(*relative_paths: str) -> bool:
    return all((REPO_ROOT / p.lstrip("./")).is_file() for p in relative_paths)


# Single tuple per case so fixture `params=` receives one value
TORCH_WEIGHT_SETS = [
    pytest.param(
        (
            "./edm_moi_chembl_15_39.pt",
            "./adj_mat_seer_chembl_15_39.pt",
            "full",
        ),
        id="full",
    ),
    pytest.param(
        (
            "./small_edm_moi_chembl_15_39.pt",
            "./small_adj_mat_seer_obabel_15_39.pt",
            "distilled",
        ),
        id="distilled",
        marks=pytest.mark.skipif(
            not _weights_present(
                "small_edm_moi_chembl_15_39.pt",
                "small_adj_mat_seer_obabel_15_39.pt",
            ),
            reason="distilled torch weights missing",
        ),
    ),
]

ONNX_WEIGHT_SETS = [
    pytest.param(
        (
            "./egnn_chembl_15_39.onnx",
            "./adj_mat_seer_chembl_15_39.onnx",
            "full",
        ),
        id="full",
    ),
    pytest.param(
        (
            "./small_egnn_chembl_15_39.onnx",
            "./small_adj_mat_seer_obabel_15_39.onnx",
            "distilled",
        ),
        id="distilled",
        marks=pytest.mark.skipif(
            not _weights_present(
                "small_egnn_chembl_15_39.onnx",
                "small_adj_mat_seer_obabel_15_39.onnx",
            ),
            reason="distilled onnx weights missing",
        ),
    ),
]


@pytest.fixture(scope="session")
def paba_mol():
    mol = Chem.MolFromMolFile("assets/demo_files/paba.mol", removeHs=False)
    assert mol is not None, "Failed to load paba.mol"
    return mol


@pytest.fixture
def paba_mol_no_hs(paba_mol):
    return Chem.RemoveHs(paba_mol)


@pytest.fixture(scope="session")
def paba_coords(paba_mol):
    mol_no_hs = Chem.RemoveHs(paba_mol)
    conf = mol_no_hs.GetConformer()
    return torch.tensor(conf.GetPositions(), dtype=torch.float32)


@pytest.fixture(scope="session")
def simple_coords():
    return torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


@pytest.fixture(scope="session")
def batch_coords():
    torch.manual_seed(42)
    return torch.randn(2, 5, 3)


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def atom_decoder():
    return ATOM_DECODER


@pytest.fixture(scope="session")
def context_norms():
    return {k: torch.tensor(v) for k, v in CONTEXT_NORMS.items()}
