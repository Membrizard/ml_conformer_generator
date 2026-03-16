import pytest
from pathlib import Path

from rdkit import Chem, RDLogger
from src.mlconfgen import (
    MLConformerGenerator,
    MLConformerGeneratorONNX,
    evaluate_samples,
)
from src.mlconfgen.utils import extract_fragment, align_mol_to_principal_frame, set_conformer_positions
from onnx_export import export_to_onnx

RDLogger.DisableLog("rdApp.*")


@pytest.fixture(scope="module")
def diffusion_steps():
    return 50


@pytest.fixture(scope="module")
def generator(device, diffusion_steps):
    generator = MLConformerGeneratorONNX(
        egnn_onnx="./egnn_chembl_15_39.onnx",
        adj_mat_seer_onnx="./adj_mat_seer_chembl_15_39.onnx",
        diffusion_steps=diffusion_steps,
    )
    return generator


@pytest.fixture(scope="module")
def ceyyag():
    return Chem.MolFromMolFile("./assets/demo_files/ceyyag.mol")


@pytest.fixture(scope="module")
def pif_aligned_ceyyag():
    ref_mol = Chem.MolFromMolFile("./assets/demo_files/ceyyag.mol")
    ref_mol = Chem.RemoveHs(ref_mol)
    context, _, _, aligned_coord = align_mol_to_principal_frame(ref_mol)
    aligned_mol = set_conformer_positions(ref_mol, aligned_coord)

    return aligned_mol


@pytest.fixture(scope="module")
def ref_context():
    mol = Chem.MolFromMolFile("./assets/demo_files/ceyyag.mol")
    mol = Chem.RemoveHs(mol)
    context, _, _, _ = align_mol_to_principal_frame(mol)
    return context.detach().numpy()


@pytest.mark.slow
def test_onnx_export():
    torch_generator = MLConformerGenerator(
        edm_weights="./edm_moi_chembl_15_39.pt",
        adj_mat_seer_weights="./adj_mat_seer_chembl_15_39.pt",
        diffusion_steps=100,
    )

    export_to_onnx(model=torch_generator)

    edm_path = Path("./egnn_chembl_15_39.onnx")
    adj_path = Path("./adj_mat_seer_chembl_15_39.onnx")

    for path in [edm_path, adj_path]:
        assert path.exists(), f"Missing file: {path}"
        assert path.is_file(), f"Not a file: {path}"


@pytest.mark.slow
def test_basic_generation_ref_mol_onnx(generator, ceyyag):
    n_samples = 20
    samples = generator.generate_conformers(
        reference_conformer=ceyyag,
        n_samples=n_samples,
        variance=1,
        resample_steps=0,
    )

    _, std_samples = evaluate_samples(ceyyag, samples)

    valid_samples = len(std_samples) / n_samples
    assert valid_samples > 0.3

    average_shape_similarity = 0
    for sample in std_samples:
        average_shape_similarity += round(sample["shape_tanimoto"], 2)
    average_shape_similarity = average_shape_similarity / len(std_samples)
    assert average_shape_similarity > 0.3


@pytest.mark.slow
def test_basic_generation_ref_context_onnx(generator, ref_context):
    n_samples = 20
    samples = generator.generate_conformers(
        reference_context=ref_context,
        n_atoms=17,
        n_samples=n_samples,
        variance=1,
    )

    valid_samples = len(samples) / n_samples
    assert valid_samples > 0.3


@pytest.mark.slow
def test_basic_generation_ff_mol_ref_mol_onnx(generator, ceyyag):
    ff_idx = {3, 5, 6, 7, 8, 9, 10}
    fixed_fragment = extract_fragment(ceyyag, ff_idx)

    n_samples = 20

    samples = generator.generate_conformers(
        reference_conformer=ceyyag,
        n_samples=n_samples,
        variance=1,
        resample_steps=2,
        fixed_fragment=fixed_fragment,
        blend_power=3,
    )

    _, std_samples = evaluate_samples(ceyyag, samples)

    valid_samples = len(std_samples) / n_samples
    assert valid_samples >= 0.1

    average_shape_similarity = 0
    for sample in std_samples:
        average_shape_similarity += round(sample["shape_tanimoto"], 2)
    average_shape_similarity = average_shape_similarity / len(std_samples)
    assert average_shape_similarity > 0.3


@pytest.mark.slow
def test_basic_generation_ff_set_ref_mol_onnx(generator, ceyyag):
    ff_idx = {3, 5, 6, 7, 8, 9, 10}
    n_samples = 20

    samples = generator.generate_conformers(
        reference_conformer=ceyyag,
        n_samples=n_samples,
        variance=1,
        resample_steps=2,
        fixed_fragment=ff_idx,
        blend_power=3,
    )

    _, std_samples = evaluate_samples(ceyyag, samples)

    valid_samples = len(std_samples) / n_samples
    assert valid_samples >= 0.15

    average_shape_similarity = 0
    for sample in std_samples:
        average_shape_similarity += round(sample["shape_tanimoto"], 2)
    average_shape_similarity = average_shape_similarity / len(std_samples)
    assert average_shape_similarity > 0.3


@pytest.mark.slow
def test_basic_generation_ff_set_ref_context_onnx(generator, ref_context):
    ff_idx = {3, 5, 6, 7, 8, 9, 10}
    with pytest.raises(
        ValueError,
        match="'fixed_fragment' must be a Mol object when generating from a reference context.",
    ):
        generator.generate_conformers(
            reference_context=ref_context,
            n_atoms=17,
            n_samples=20,
            variance=1,
            resample_steps=4,
            fixed_fragment=ff_idx,
            blend_power=3,
        )


@pytest.mark.slow
def test_basic_generation_ff_mol_ref_context_onnx(generator, pif_aligned_ceyyag, ref_context):
    ff_idx = {3, 5, 6, 7, 8, 9, 10}

    fixed_fragment = extract_fragment(pif_aligned_ceyyag, ff_idx)
    n_samples = 20

    samples = generator.generate_conformers(
        reference_context=ref_context,
        n_atoms=17,
        n_samples=n_samples,
        variance=1,
        resample_steps=4,
        fixed_fragment=fixed_fragment,
        blend_power=3,
    )

    valid_samples = len(samples) / n_samples
    assert valid_samples >= 0.15
