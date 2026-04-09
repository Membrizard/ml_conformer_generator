import rdkit.Chem
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

from ..utils import set_conformer_positions
from .shape_similarity import (best_pi_rotation_by_tanimoto,
                               get_shape_quadrupole_for_molecule)

FP_SIZE = 2048
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=FP_SIZE, includeChirality=False, useBondTypes=True
)


def evaluate_samples(
    reference: rdkit.Chem.Mol,
    samples: list[rdkit.Chem.Mol],
    generator: rdFingerprintGenerator = GENERATOR,
    sanitize_ref: bool = True,
) -> tuple[str, list[dict]]:
    """
    Calculate chemical and shape similarity of the generated samples to reference, while ignoring Hs
    :param reference: reference mol
    :param samples: a list of generated mols
    :param generator: fingerprint generator
    :param sanitize_ref: If reference molecule should be sanitized
    :return: molblock of a reference in a principal frame, a list of sample conformers molblocks, aligned with reference,
             along with chemical and shape tanimoto scores.
    """

    # Ensure Hs are stripped off Reference
    reference = Chem.RemoveHs(reference, sanitize=sanitize_ref)

    if sanitize_ref:
        fp_ref = generator.GetFingerprint(reference)
    else:
        fp_ref = None

    conf = reference.GetConformer()
    ref_coord = torch.tensor(conf.GetPositions(), dtype=torch.float32)

    # move coord to center
    virtual_com = torch.mean(ref_coord, dim=0)
    ref_coord = ref_coord - virtual_com

    r_s_mom, sq_ref_coord = get_shape_quadrupole_for_molecule(coordinates=ref_coord)
    # Set mol object coordinates to the principal frame
    pf_reference = set_conformer_positions(reference, sq_ref_coord)
    ref_mol_block = Chem.MolToMolBlock(pf_reference)

    results = []
    for sample in samples:
        # Calculate chemical similarity Tanimoto score
        # Ensure Hydrogens are stripped off

        # Ensure Hs are stripped off Sample
        sample = Chem.RemoveHs(sample)

        fp_sample = generator.GetFingerprint(sample)

        if sanitize_ref:
            chemical_tanimoto = TanimotoSimilarity(fp_ref, fp_sample)
        else:
            chemical_tanimoto = 0

        sample_conf = sample.GetConformer()
        sample_coord = torch.tensor(sample_conf.GetPositions(), dtype=torch.float32)

        # Move Center to COM
        s_virtual_com = torch.mean(sample_coord, dim=0)
        sample_coord = sample_coord - s_virtual_com
        s_s_mom, sq_sample_coord = get_shape_quadrupole_for_molecule(
            coordinates=sample_coord
        )

        best_coord, shape_tanimoto = best_pi_rotation_by_tanimoto(
            sq_ref_coord, sq_sample_coord
        )

        aligned_sample = set_conformer_positions(sample, best_coord)

        results.append(
            {
                "mol_block": Chem.MolToMolBlock(aligned_sample),
                "shape_tanimoto": shape_tanimoto,
                "chemical_tanimoto": chemical_tanimoto,
            }
        )
    return ref_mol_block, results
