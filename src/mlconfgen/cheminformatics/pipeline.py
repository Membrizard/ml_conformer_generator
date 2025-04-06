import rdkit.Chem
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Geometry import Point3D

from .shape_similarity import (
    get_shape_quadrupole_for_molecule,
    rotate_coord,
    tanimoto_score,
)

FP_SIZE = 2048
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=FP_SIZE, includeChirality=False, useBondTypes=True
)


def evaluate_samples(
    reference: rdkit.Chem.Mol,
    samples: list[rdkit.Chem.Mol],
    generator: rdFingerprintGenerator = GENERATOR,
) -> tuple[str, list[dict]]:
    """
    Calculate chemical and shape similarity of the generated samples to reference
    :param reference: reference mol
    :param samples: a list of generated mols
    :param generator: fingerprint generator
    :return: molblock of a reference in a principal frame, a list of sample conformers molblocks, aligned with reference,
             along with chemical and shape tanimoto scores.
    """

    fp_ref = generator.GetFingerprint(reference)
    conf = reference.GetConformer()
    ref_coord = torch.tensor(conf.GetPositions(), dtype=torch.float32)

    # move coord to center
    virtual_com = torch.mean(ref_coord, dim=0)
    ref_coord = ref_coord - virtual_com

    r_s_mom, sq_ref_coord = get_shape_quadrupole_for_molecule(coordinates=ref_coord)
    # Set mol object coordinates to the principal frame
    pf_reference = set_conformer_positions(reference, sq_ref_coord)
    ref_mol_block = Chem.MolToMolBlock(pf_reference)
    # Create a mol object without Hs for shape similarity calcs
    pf_ref_no_hs = Chem.RemoveHs(pf_reference)
    conf_no_hs = pf_ref_no_hs.GetConformer()
    sq_ref_coord_no_hs = torch.tensor(conf_no_hs.GetPositions(), dtype=torch.float32)

    pi = torch.pi
    rotations = [
        torch.tensor([pi, 0, 0]),
        torch.tensor([0, pi, 0]),
        torch.tensor([0, 0, pi]),
    ]

    results = []
    for sample in samples:
        # Calculate chemical similarity Tanimoto score
        # Ensure Hydrogens are stripped off
        fp_sample = generator.GetFingerprint(sample)

        chemical_tanimoto = TanimotoSimilarity(fp_ref, fp_sample)

        sample_conf = sample.GetConformer()
        sample_coord = torch.tensor(sample_conf.GetPositions(), dtype=torch.float32)

        # Move Center to COM
        s_virtual_com = torch.mean(sample_coord, dim=0)
        sample_coord = sample_coord - s_virtual_com
        s_s_mom, sq_sample_coord = get_shape_quadrupole_for_molecule(
            coordinates=sample_coord
        )

        shape_tanimoto = tanimoto_score(sq_ref_coord_no_hs, sq_sample_coord)
        best_coord = sq_sample_coord

        # Calculate Best shape similarity Tanimoto score
        for angles in rotations:
            rot_coord = rotate_coord(coord=sq_sample_coord, angles=angles)
            score = tanimoto_score(sq_ref_coord_no_hs, rot_coord)
            if score > shape_tanimoto:
                shape_tanimoto = score
                best_coord = rot_coord

        aligned_sample = set_conformer_positions(sample, best_coord)

        results.append(
            {
                "mol_block": Chem.MolToMolBlock(aligned_sample),
                "shape_tanimoto": shape_tanimoto,
                "chemical_tanimoto": chemical_tanimoto,
            }
        )
    return ref_mol_block, results


def set_conformer_positions(mol, coord):
    conf = mol.GetConformer()
    for i, point in enumerate(coord):
        x, y, z = point.tolist()
        conf.SetAtomPosition(i, Point3D(x, y, z))

    return mol
