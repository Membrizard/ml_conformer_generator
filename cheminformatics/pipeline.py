import torch

from .shape_similarity import get_shape_quadrupole_for_molecule

def characterise_samples(reference, samples):
    conf = reference.GetConformer()
    ref_coord = torch.tensor(conf.GetPositions(), dtype=torch.float32)

    # move coord to center
    virtual_com = torch.mean(ref_coord, dim=0)
    ref_coord = ref_coord - virtual_com

    r_s_mom, sq_ref_coord = get_shape_quadrupole_for_molecule(coordinates=ref_coord,
                                                              amplitude=p,
                                                              generic_atom_radius=atom_radius,
                                                              n_terms=6,
                                                              neighbour_threshold=2 * atom_radius)
    return None
