from rdkit import Chem

from mlconfgen.cheminformatics.pipeline import evaluate_samples


def test_evaluate_samples_returns_tuple(paba_mol_no_hs):
    ref_block, results = evaluate_samples(paba_mol_no_hs, [paba_mol_no_hs])
    assert isinstance(ref_block, str)
    assert isinstance(results, list)


def test_evaluate_samples_ref_molblock_valid(paba_mol_no_hs):
    ref_block, _ = evaluate_samples(paba_mol_no_hs, [paba_mol_no_hs])
    mol = Chem.MolFromMolBlock(ref_block)
    assert mol is not None


def test_evaluate_samples_result_keys(paba_mol_no_hs):
    _, results = evaluate_samples(paba_mol_no_hs, [paba_mol_no_hs])
    assert len(results) == 1
    d = results[0]
    assert "mol_block" in d
    assert "shape_tanimoto" in d
    assert "chemical_tanimoto" in d


def test_evaluate_samples_self_similarity(paba_mol_no_hs):
    _, results = evaluate_samples(paba_mol_no_hs, [paba_mol_no_hs])
    assert results[0]["chemical_tanimoto"] == 1.0


def test_evaluate_samples_shape_tanimoto_range(paba_mol_no_hs):
    _, results = evaluate_samples(paba_mol_no_hs, [paba_mol_no_hs])
    score = results[0]["shape_tanimoto"]
    assert 0.0 <= score <= 1.0 + 0.01


def test_evaluate_samples_empty_list(paba_mol_no_hs):
    ref_block, results = evaluate_samples(paba_mol_no_hs, [])
    assert isinstance(ref_block, str)
    assert results == []
