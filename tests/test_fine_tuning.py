import os.path
import shutil
from pathlib import Path

import pytest
import torch
from rdkit import Chem, RDLogger

from onnx_export import export_to_onnx
from src.mlconfgen import (MLConformerGenerator, MLConformerGeneratorONNX,
                           evaluate_samples)
from src.mlconfgen.rl_fine_tuning.edm_adapter import EDMAdapter

RDLogger.DisableLog("rdApp.*")


@pytest.fixture(scope="module")
def diffusion_steps():
    return 10


@pytest.fixture(scope="module")
def device():
    if torch.cuda.is_available():
        _device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        _device = torch.device("mps:0")
    else:
        _device = torch.device("cpu")
    return _device


@pytest.fixture(scope="module")
def ifm_device():
    if torch.cuda.is_available():
        _device = torch.device("cuda:0")
    else:
        _device = torch.device("cpu")
    return _device


@pytest.fixture(scope="module")
def generator(device, diffusion_steps):
    generator = MLConformerGenerator(
        edm_weights="./edm_moi_chembl_15_39.pt",
        adj_mat_seer_weights="./adj_mat_seer_chembl_15_39.pt",
        device=device,
        diffusion_steps=diffusion_steps,
    )
    return generator


@pytest.fixture(scope="module")
def ceyyag():
    return Chem.MolFromMolFile("./assets/demo_files/ceyyag.mol")


@pytest.fixture(scope="module")
def artifacts_dir():
    dir_path = "./test_rl_fine_tuning"
    yield dir_path
    shutil.rmtree(dir_path)


@pytest.fixture(scope="module")
def latest_checkpoint(artifacts_dir):
    return f"{artifacts_dir}/latest_checkpoint.pt"


@pytest.fixture(scope="module")
def onnx_paths(artifacts_dir):
    return (
            Path(f"{artifacts_dir}/egnn_chembl_15_39.onnx"),
            Path(f"{artifacts_dir}/adj_mat_seer_chembl_15_39.onnx"),
            Path(f"{artifacts_dir}/finetune_checkpoint.onnx"),
            )


@pytest.fixture(scope="module")
def generator_onnx(device, diffusion_steps, onnx_paths):
    edm, adj, ft = onnx_paths
    generator = MLConformerGeneratorONNX(
        egnn_onnx=edm,
        adj_mat_seer_onnx=adj,
        finetune_checkpoint_onnx=ft,
        diffusion_steps=diffusion_steps,
    )
    return generator


@pytest.mark.slow
def test_basic_fine_tuning(generator, ceyyag, artifacts_dir, latest_checkpoint):
    generator.fine_tune(
        scoring_function=None,
        reference_conformer=ceyyag,
        variance=1,
        n_epochs=2,
        train_batch_size=16,
        eval_batch_size=16,
        lambda_edm_adapter=1.5,
        lambda_edm_reg=0.2,
        learning_rate=8e-5,
        sigma=60.0,
        temperature=1.5,
        n_samples_per_mol=8,
        reward_clip=(-1.0, 1.0),
        eval_every=2,
        save_dir=artifacts_dir,
    )

    # Make sure checkpoints are saved
    assert os.path.isfile(latest_checkpoint)
    generator.load_finetune_checkpoint(latest_checkpoint)
    assert generator.edm_adapter is not None
    assert isinstance(generator.edm_adapter, EDMAdapter)


@pytest.mark.slow
def test_finetuned_generation_torch(generator, ceyyag, latest_checkpoint):
    generator.load_finetune_checkpoint(latest_checkpoint)
    n_samples = 20
    samples = generator.generate_conformers(
        reference_conformer=ceyyag,
        n_samples=n_samples,
        variance=1,
        resample_steps=0,
    )

    _, std_samples = evaluate_samples(ceyyag, samples)

    valid_samples = len(std_samples) / n_samples
    assert valid_samples > 0.1

    average_shape_similarity = 0
    for sample in std_samples:
        average_shape_similarity += round(sample["shape_tanimoto"], 2)
    average_shape_similarity = average_shape_similarity / len(std_samples)
    assert average_shape_similarity > 0.1


@pytest.mark.slow
def test_finetuned_onnx_export(generator, latest_checkpoint, onnx_paths):
    generator.load_finetune_checkpoint(latest_checkpoint)
    edm_path, adj_path, edm_adapter_checkpoint = onnx_paths

    export_to_onnx(
        model=generator,
        egnn_save_path=edm_path,
        adj_mat_seer_save_path=adj_path,
        edm_adapter_save_path=edm_adapter_checkpoint,
    )

    for path in [edm_path, adj_path, edm_adapter_checkpoint]:
        assert path.exists(), f"Missing file: {path}"
        assert path.is_file(), f"Not a file: {path}"


@pytest.mark.slow
def test_finetuned_generation_onnx(generator_onnx, ceyyag):
    n_samples = 20
    samples = generator_onnx.generate_conformers(
        reference_conformer=ceyyag,
        n_samples=n_samples,
        variance=1,
        resample_steps=0,
    )

    _, std_samples = evaluate_samples(ceyyag, samples)

    valid_samples = len(std_samples) / n_samples
    assert valid_samples > 0.1

    average_shape_similarity = 0
    for sample in std_samples:
        average_shape_similarity += round(sample["shape_tanimoto"], 2)
    average_shape_similarity = average_shape_similarity / len(std_samples)
    assert average_shape_similarity > 0.1
