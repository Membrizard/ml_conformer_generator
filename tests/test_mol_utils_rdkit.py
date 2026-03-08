import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from src.mlconfgen.utils.config import DIMENSION, MIN_N_NODES, PERMITTED_ELEMENTS
from src.mlconfgen.utils.mol_utils import (
    align_mol_to_principal_frame,
    canonicalise,
    ifm_get_xh_from_fragment,
    ifm_prepare_fragments_for_merge,
    ifm_prepare_gen_fragment_context,
    prepare_adj_mat_seer_input,
    prepare_fragment,
    redefine_bonds,
    samples_to_rdkit_mol,
    set_conformer_positions,
)
from mlconfgen.utils.molgraph import MolGraph


# --- samples_to_rdkit_mol ---


def test_samples_to_mol_returns_list(paba_mol_no_hs, atom_decoder):
    g = MolGraph.from_mol(paba_mol_no_hs, remove_hs=False)
    n = len(g.x)
    oh = g.one_hot_elements_encoding(max_n_nodes=n)
    conf = paba_mol_no_hs.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    # Add batch dim
    pos = pos.unsqueeze(0)
    oh = oh.unsqueeze(0).float()
    node_mask = torch.ones(1, n, 1)
    result = samples_to_rdkit_mol(pos, oh, node_mask, atom_decoder)
    assert isinstance(result, list)
    assert len(result) > 0
    assert isinstance(result[0], Chem.Mol)


def test_samples_to_mol_atom_count(paba_mol_no_hs, atom_decoder):
    g = MolGraph.from_mol(paba_mol_no_hs, remove_hs=False)
    n = len(g.x)
    oh = g.one_hot_elements_encoding(max_n_nodes=n)
    conf = paba_mol_no_hs.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    pos = pos.unsqueeze(0)
    oh = oh.unsqueeze(0).float()
    node_mask = torch.ones(1, n, 1)
    result = samples_to_rdkit_mol(pos, oh, node_mask, atom_decoder)
    assert result[0].GetNumAtoms() == n


def test_samples_to_mol_no_mask(paba_mol_no_hs, atom_decoder):
    g = MolGraph.from_mol(paba_mol_no_hs, remove_hs=False)
    n = len(g.x)
    oh = g.one_hot_elements_encoding(max_n_nodes=n)
    conf = paba_mol_no_hs.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    pos = pos.unsqueeze(0)
    oh = oh.unsqueeze(0).float()
    result = samples_to_rdkit_mol(pos, oh, node_mask=None, atom_decoder=atom_decoder)
    assert isinstance(result, list)


# --- canonicalise ---


def test_canonicalise_returns_mol(paba_mol_no_hs):
    mol = canonicalise(paba_mol_no_hs)
    assert isinstance(mol, Chem.Mol)


def test_canonicalise_atom_count_preserved(paba_mol_no_hs):
    mol = canonicalise(paba_mol_no_hs)
    assert mol.GetNumAtoms() == paba_mol_no_hs.GetNumAtoms()


# --- redefine_bonds ---


def test_redefine_bonds_returns_mol(paba_mol_no_hs):
    g = MolGraph.from_mol(paba_mol_no_hs, remove_hs=False)
    adj = g.adjacency_matrix(padded=True)
    mol = redefine_bonds(paba_mol_no_hs, adj)
    assert isinstance(mol, Chem.Mol)


def test_redefine_bonds_atom_count(paba_mol_no_hs):
    g = MolGraph.from_mol(paba_mol_no_hs, remove_hs=False)
    adj = g.adjacency_matrix(padded=True)
    mol = redefine_bonds(paba_mol_no_hs, adj)
    assert mol.GetNumAtoms() == paba_mol_no_hs.GetNumAtoms()


# --- set_conformer_positions ---


def test_set_positions_updates(paba_mol_no_hs):
    mol = Chem.RWMol(paba_mol_no_hs)
    n = mol.GetNumAtoms()
    new_coords = torch.randn(n, 3)
    updated = set_conformer_positions(mol, new_coords)
    conf = updated.GetConformer()
    for i in range(n):
        pos = conf.GetAtomPosition(i)
        assert abs(pos.x - new_coords[i, 0].item()) < 1e-4
        assert abs(pos.y - new_coords[i, 1].item()) < 1e-4
        assert abs(pos.z - new_coords[i, 2].item()) < 1e-4


# --- prepare_adj_mat_seer_input ---


def test_ams_input_shapes(paba_mol_no_hs, device):
    elements, dist, adj, mols = prepare_adj_mat_seer_input(
        [paba_mol_no_hs], DIMENSION, device
    )
    assert elements.shape == (1, DIMENSION)
    assert dist.shape == (1, DIMENSION, DIMENSION)
    assert adj.shape == (1, DIMENSION, DIMENSION)
    assert len(mols) == 1


# --- ifm_get_xh_from_fragment ---


def test_xh_fragment_shapes(paba_mol, device):
    x, h = ifm_get_xh_from_fragment(paba_mol, device)
    n_heavy = Chem.RemoveAllHs(paba_mol).GetNumAtoms()
    assert x.shape == (n_heavy, 3)
    assert h.shape == (n_heavy, len(PERMITTED_ELEMENTS))


# --- prepare_fragment ---


def test_fragment_shapes(device):
    # Create a small fragment (3 atoms)
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    from rdkit.Chem import AllChem
    AllChem.EmbedMolecule(mol, randomSeed=42)

    z_known, fixed_mask = prepare_fragment(
        n_samples=2,
        fixed_fragment=mol,
        device=device,
        max_n_nodes=DIMENSION,
        min_n_nodes=MIN_N_NODES,
    )
    assert z_known.shape == (2, DIMENSION, 3 + len(PERMITTED_ELEMENTS))
    assert fixed_mask.shape == (2, DIMENSION, 1)


def test_fragment_too_large_raises(device):
    # Create fragment with >= MIN_N_NODES heavy atoms
    smi = "C" * MIN_N_NODES
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    from rdkit.Chem import AllChem
    AllChem.EmbedMolecule(mol, randomSeed=42)

    with pytest.raises(ValueError):
        prepare_fragment(
            n_samples=1,
            fixed_fragment=mol,
            device=device,
            max_n_nodes=DIMENSION,
            min_n_nodes=MIN_N_NODES,
        )


# --- ifm_prepare_gen_fragment_context ---


def test_gen_fragment_context_shapes(device, context_norms):
    # Use a small fragment (< MIN_N_NODES heavy atoms) to satisfy the size constraint
    small_mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(small_mol, randomSeed=42)
    x, h = ifm_get_xh_from_fragment(small_mol, device)
    n_heavy = x.size(0)
    ref_context = torch.tensor([100.0, 400.0, 500.0])
    n_nodes = torch.tensor([20, 25])

    frag_node_mask, frag_edge_mask, ctx, shift, rotation = ifm_prepare_gen_fragment_context(
        fixed_fragment_x=x,
        reference_context=ref_context,
        context_norms=context_norms,
        n_nodes=n_nodes,
        max_n_nodes=DIMENSION,
        min_n_nodes=MIN_N_NODES,
        device=device,
    )
    max_frag = DIMENSION - n_heavy
    assert frag_node_mask.shape == (2, max_frag, 1)
    assert shift.shape == (2, 3)
    assert rotation.shape == (2, 3, 3)


# --- ifm_prepare_fragments_for_merge ---


def test_merge_fragments_shapes(paba_mol, device):
    x, h = ifm_get_xh_from_fragment(paba_mol, device)
    n_heavy = x.size(0)
    B = 2
    gen_x = torch.randn(B, 5, 3)
    gen_h = torch.randn(B, 5, len(PERMITTED_ELEMENTS))

    z_known, fixed_mask = ifm_prepare_fragments_for_merge(
        fixed_fragment_x=x,
        fixed_fragment_h=h,
        gen_fragments_x=gen_x,
        gen_fragments_h=gen_h,
        device=device,
        max_n_nodes=DIMENSION,
    )
    total_atoms = n_heavy + 5
    assert z_known.shape == (B, total_atoms, 3 + len(PERMITTED_ELEMENTS))
    assert fixed_mask.shape == (B, DIMENSION, 1)


# --- align_mol_to_principal_frame ---


def test_align_mol_to_principal_frame(paba_mol_no_hs):
    context, shift, rotation, aligned_coord = align_mol_to_principal_frame(paba_mol_no_hs)
    n_atoms = paba_mol_no_hs.GetNumAtoms()

    assert context.shape == (3,)
    assert shift.shape == (3,)
    assert rotation.shape == (3, 3)
    assert aligned_coord.shape == (n_atoms, 3)

    # Rotation matrix is orthogonal
    assert torch.allclose(rotation.T @ rotation, torch.eye(3), atol=1e-5)

    # Coordinates are centered (COM near zero)
    assert torch.allclose(aligned_coord.mean(dim=0), torch.zeros(3), atol=1e-4)
