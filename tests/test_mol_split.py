import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from src.mlconfgen.utils.mol_split import (
    extract_fragment,
    find_cuttable_bonds,
    split_molecule_size_constrained,
)


# --- Helpers ---


@pytest.fixture(scope="module")
def biphenyl():
    """Biphenyl: 12 heavy atoms, single bond connecting two rings."""
    mol = Chem.MolFromSmiles("c1ccc(-c2ccccc2)cc1")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


@pytest.fixture(scope="module")
def long_chain():
    """Hexadecane: 16 heavy atoms, many cuttable bonds."""
    mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


@pytest.fixture(scope="module")
def benzene():
    """Benzene: 6 heavy atoms, no cuttable bonds between heavy atoms (all ring bonds)."""
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


@pytest.fixture(scope="module")
def benzene_no_hs():
    """Benzene without Hs: only ring bonds."""
    mol = Chem.MolFromSmiles("c1ccccc1")
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


# --- find_cuttable_bonds ---


def test_cuttable_bonds_ring_only(benzene_no_hs):
    """On a ring-only molecule without Hs, no bonds are cuttable."""
    bonds = find_cuttable_bonds(benzene_no_hs)
    assert len(bonds) == 0


def test_cuttable_bonds_biphenyl(biphenyl):
    bonds = find_cuttable_bonds(biphenyl)
    assert len(bonds) >= 1


def test_cuttable_bonds_chain(long_chain):
    bonds = find_cuttable_bonds(long_chain)
    assert len(bonds) >= 10


# --- split_molecule_size_constrained ---


def test_split_small_mol_single_fragment(benzene):
    """Molecule with <= min_size heavy atoms returns (fragments, mols) early."""
    frags, frag_mols = split_molecule_size_constrained(benzene, min_size=6, max_size=20)
    assert len(frags) == 1
    assert len(frags[0]) == 6
    assert len(frag_mols) == 1


def test_split_respects_size_window(long_chain):
    """All fragments should fall within the size window."""
    min_s, max_s = 6, 10
    frags = split_molecule_size_constrained(long_chain, min_size=min_s, max_size=max_s)
    for frag in frags:
        assert min_s <= len(frag) <= max_s


def test_split_covers_all_heavy_atoms(long_chain):
    """Union of fragments should equal the full set of heavy atoms."""
    frags = split_molecule_size_constrained(long_chain, min_size=6, max_size=10)
    all_atoms = set()
    for frag in frags:
        all_atoms |= frag
    heavy = {a.GetIdx() for a in long_chain.GetAtoms() if a.GetAtomicNum() > 1}
    assert all_atoms == heavy


def test_split_fragments_disjoint(long_chain):
    """Fragments should not overlap."""
    frags = split_molecule_size_constrained(long_chain, min_size=6, max_size=10)
    seen = set()
    for frag in frags:
        assert seen.isdisjoint(frag)
        seen |= frag


def test_split_biphenyl(biphenyl):
    """Biphenyl (12 heavy atoms) with max_size=8 should produce 2 fragments."""
    frags = split_molecule_size_constrained(biphenyl, min_size=5, max_size=8)
    assert len(frags) == 2


def test_split_no_cuttable_bonds(benzene):
    """No cuttable bonds → single fragment even if above max_size."""
    frags = split_molecule_size_constrained(benzene, min_size=2, max_size=4)
    assert len(frags) == 1


# --- extract_fragment ---


def test_extract_fragment_atom_count(biphenyl):
    """Extracted fragment should have the requested number of atoms."""
    indices = [0, 1, 2, 3, 4, 5]
    frag = extract_fragment(biphenyl, indices)
    assert frag.GetNumAtoms() == len(indices)


def test_extract_fragment_preserves_coords(biphenyl):
    """Coordinates in the fragment should match the original molecule."""
    indices = [0, 1, 2]
    frag = extract_fragment(biphenyl, indices)
    orig_conf = biphenyl.GetConformer()
    frag_conf = frag.GetConformer()
    for new_i, old_i in enumerate(sorted(indices)):
        orig_pos = orig_conf.GetAtomPosition(old_i)
        frag_pos = frag_conf.GetAtomPosition(new_i)
        assert abs(orig_pos.x - frag_pos.x) < 1e-4
        assert abs(orig_pos.y - frag_pos.y) < 1e-4
        assert abs(orig_pos.z - frag_pos.z) < 1e-4


def test_extract_fragment_preserves_bonds(biphenyl):
    """Bonds connecting kept atoms should be preserved."""
    # First 6 atoms of biphenyl form a ring
    indices = list(range(6))
    frag = extract_fragment(biphenyl, indices)
    assert frag.GetNumBonds() > 0


def test_extract_fragment_out_of_range_raises(biphenyl):
    """Out-of-range atom index should raise IndexError."""
    bad_indices = [0, 1, biphenyl.GetNumAtoms() + 10]
    with pytest.raises(IndexError):
        extract_fragment(biphenyl, bad_indices)
