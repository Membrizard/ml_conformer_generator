import pytest
import torch
from rdkit import Chem, RDLogger

from src.mlconfgen import (MLConformerGenerator, evaluate_samples,
                           ff_inertial_fragment_matching,
                           inertial_fragment_matching)
from src.mlconfgen.utils import (align_mol_to_principal_frame,
                                 extract_fragment, set_conformer_positions)

RDLogger.DisableLog("rdApp.*")


@pytest.fixture(scope="module")
def diffusion_steps():
    return 50


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
def ifm_generator(ifm_device, diffusion_steps):
    generator = MLConformerGenerator(
        edm_weights="./edm_moi_chembl_6_39_fragments.pt",
        adj_mat_seer_weights="./adj_mat_seer_chembl_15_39.pt",
        device=ifm_device,
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
    return context


@pytest.mark.slow
def test_basic_generation_ref_mol(generator, ceyyag):
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
def test_basic_generation_ref_context(generator, ref_context):
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
def test_basic_generation_ff_mol_ref_mol(generator, ceyyag):
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
def test_basic_generation_ff_set_ref_mol(generator, ceyyag):
    ff_idx = {3, 5, 6, 7, 8, 9, 10}
    n_samples = 20

    samples = generator.generate_conformers(
        reference_conformer=ceyyag,
        n_samples=n_samples,
        variance=1,
        resample_steps=4,
        fixed_fragment=ff_idx,
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
def test_basic_generation_ff_set_ref_context(generator, ref_context):
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
def test_basic_generation_ff_mol_ref_context(
    generator, pif_aligned_ceyyag, ref_context
):
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
    assert valid_samples >= 0.1


@pytest.mark.slow
def test_ifm(ifm_generator, ceyyag):
    n_samples = 20
    samples = inertial_fragment_matching(
        reference_conformer=ceyyag,
        n_samples=n_samples,
        generator=ifm_generator,
        variance=1,
        predict_bonds=True,
        verbose=False,
    )

    _, std_samples = evaluate_samples(ceyyag, samples)

    valid_samples = len(std_samples) / n_samples
    assert valid_samples > 0.3

    average_shape_similarity = 0
    for sample in std_samples:
        average_shape_similarity += round(sample["shape_tanimoto"], 2)
    assert average_shape_similarity > 0.3


@pytest.mark.slow
def test_ifm_ff_mol_ref_mol(ifm_generator, generator, ceyyag):
    ff_idx = {3, 5, 6, 7, 8, 9, 10}
    fixed_fragment = extract_fragment(ceyyag, ff_idx)

    n_samples = 20
    samples = ff_inertial_fragment_matching(
        fixed_fragment=fixed_fragment,
        reference_conformer=ceyyag,
        generator=ifm_generator,
        merger=generator,
        n_samples=n_samples,
        variance=1,
        predict_bonds=True,
        optimize_geometry=True,
    )

    _, std_samples = evaluate_samples(ceyyag, samples)

    valid_samples = len(std_samples) / n_samples
    assert valid_samples > 0.3

    average_shape_similarity = 0
    for sample in std_samples:
        average_shape_similarity += round(sample["shape_tanimoto"], 2)
    assert average_shape_similarity > 0.3


@pytest.mark.slow
def test_ifm_ff_set_ref_mol(ifm_generator, generator, ceyyag):
    ff_idx = {3, 5, 6, 7, 8, 9, 10}
    n_samples = 20

    samples = ff_inertial_fragment_matching(
        fixed_fragment=ff_idx,
        reference_conformer=ceyyag,
        generator=ifm_generator,
        merger=generator,
        n_samples=n_samples,
        variance=1,
        predict_bonds=True,
        optimize_geometry=True,
    )

    _, std_samples = evaluate_samples(ceyyag, samples)

    valid_samples = len(std_samples) / n_samples
    assert valid_samples > 0.3

    average_shape_similarity = 0
    for sample in std_samples:
        average_shape_similarity += round(sample["shape_tanimoto"], 2)
    assert average_shape_similarity > 0.3


@pytest.mark.slow
def test_ifm_ff_set_ref_context(ifm_generator, generator, ref_context):
    ff_idx = {3, 5, 6, 7, 8, 9, 10}

    with pytest.raises(
        ValueError,
        match="'fixed_fragment' must be a Mol object when generating from a reference context.",
    ):
        ff_inertial_fragment_matching(
            fixed_fragment=ff_idx,
            reference_context=ref_context,
            n_atoms=17,
            generator=ifm_generator,
            merger=generator,
            n_samples=20,
            variance=1,
            predict_bonds=True,
            optimize_geometry=True,
        )


@pytest.mark.slow
def test_ifm_ff_mol_ref_context(
    ifm_generator, generator, pif_aligned_ceyyag, ref_context
):
    ff_idx = {3, 5, 6, 7, 8, 9, 10}
    fixed_fragment = extract_fragment(pif_aligned_ceyyag, ff_idx)

    n_samples = 20
    samples = ff_inertial_fragment_matching(
        fixed_fragment=fixed_fragment,
        reference_context=ref_context,
        n_atoms=17,
        generator=ifm_generator,
        merger=generator,
        n_samples=n_samples,
        variance=1,
        predict_bonds=True,
        optimize_geometry=True,
    )

    valid_samples = len(samples) / n_samples
    assert valid_samples > 0.3
