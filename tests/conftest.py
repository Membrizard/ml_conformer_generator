import pytest
import torch
from rdkit import Chem

from mlconfgen.utils.config import ATOM_DECODER, CONTEXT_NORMS


@pytest.fixture(scope="session")
def paba_mol():
    mol = Chem.MolFromMolFile("assets/demo_files/paba.mol", removeHs=False)
    assert mol is not None, "Failed to load paba.mol"
    return mol


@pytest.fixture(scope="session")
def paba_mol_no_hs(paba_mol):
    return Chem.RemoveHs(paba_mol)


@pytest.fixture(scope="session")
def paba_coords(paba_mol_no_hs):
    conf = paba_mol_no_hs.GetConformer()
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
