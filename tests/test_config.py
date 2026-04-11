from src.mlconfgen.utils.config import (ATOM_DECODER, CONTEXT_NORMS, DIMENSION,
                                        MAX_N_NODES, MIN_N_NODES,
                                        NUM_BOND_TYPES, PERMITTED_ELEMENTS)


def test_dimension_value():
    assert DIMENSION == 42


def test_num_bond_types_value():
    assert NUM_BOND_TYPES == 5


def test_min_max_nodes_ordering():
    assert MIN_N_NODES > 0
    assert MAX_N_NODES > 0
    assert MIN_N_NODES < MAX_N_NODES


def test_context_norms_keys():
    assert "mean" in CONTEXT_NORMS
    assert "mad" in CONTEXT_NORMS
    assert len(CONTEXT_NORMS["mean"]) == 3
    assert len(CONTEXT_NORMS["mad"]) == 3


def test_atom_decoder_completeness():
    assert len(ATOM_DECODER) == 8
    for v in ATOM_DECODER.values():
        assert isinstance(v, str)


def test_permitted_elements_tuple():
    assert isinstance(PERMITTED_ELEMENTS, tuple)
    assert len(PERMITTED_ELEMENTS) == 8
    for e in PERMITTED_ELEMENTS:
        assert isinstance(e, int) and e > 0
