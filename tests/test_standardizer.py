import pytest
from rdkit import Chem

from mlconfgen.utils.standardizer import (
    flatten_tartrate_mol,
    md_minimize_energy,
    standardize_mol,
)


def test_flatten_tartrate_no_tartrate(paba_mol):
    result = flatten_tartrate_mol(paba_mol)
    assert result.GetNumAtoms() == paba_mol.GetNumAtoms()


def test_md_minimize_returns_tuple(paba_mol):
    mol = Chem.RWMol(paba_mol)
    # Remove Hs first since md_minimize_energy will add them back
    mol = Chem.RemoveHs(mol)
    result = md_minimize_energy(mol)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], Chem.Mol)


def test_md_minimize_preserves_atoms(paba_mol):
    mol = Chem.RemoveHs(Chem.RWMol(paba_mol))
    n_before = mol.GetNumAtoms()
    result_mol, _ = md_minimize_energy(mol)
    assert result_mol.GetNumAtoms() == n_before


def test_standardize_mol_returns_mol(paba_mol):
    mol = Chem.RWMol(paba_mol)
    result = standardize_mol(mol, optimize_geometry=False)
    assert isinstance(result, Chem.Mol)


def test_standardize_mol_none_on_bad_input():
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom(6))
    mol.AddAtom(Chem.Atom(6))
    result = standardize_mol(mol.GetMol(), optimize_geometry=True)
    assert result is None


@pytest.mark.slow
def test_standardize_mol_with_geometry_opt(paba_mol):
    mol = Chem.RWMol(paba_mol)
    result = standardize_mol(mol, optimize_geometry=True)
    assert result is not None
    assert isinstance(result, Chem.Mol)
