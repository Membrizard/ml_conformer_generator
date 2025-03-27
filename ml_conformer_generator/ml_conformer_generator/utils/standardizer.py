#
#  Based on ChEMBL_StructurePipeline project
#  Copyright (c) 2019 Greg Landrum
#  All rights reserved.
#
#  This file is based on a part of the ChEMBL_StructurePipeline project.
#  The contents are covered by the terms of the MIT license
#  which is included in the file LICENSE, found at the root
#  of the source tree.


from typing import Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

# derived from the MolVS set, with ChEMBL-specific additions
_normalization_transforms = """
//	Name	SMIRKS
Nitro to N+(O-)=O	[N;X3:1](=[O:2])=[O:3]>>[*+1:1]([*-1:2])=[*:3]
Diazonium N	[*:1]-[N;X2:2]#[N;X1:3]>>[*:1]-[*+1:2]#[*:3]
Quaternary N	[N;X4;v4;+0:1]>>[*+1:1]
Trivalent O	[*:1]=[O;X2;v3;+0:2]-[#6:3]>>[*:1]=[*+1:2]-[*:3]
Sulfoxide to -S+(O-)	[!O:1][S+0;D3:2](=[O:3])[!O:4]>>[*:1][S+1:2]([O-:3])[*:4]
// this form addresses a pathological case that came up a few times in testing:
Sulfoxide to -S+(O-) 2	[!O:1][SH1+1;D3:2](=[O:3])[!O:4]>>[*:1][S+1:2]([O-:3])[*:4]
Trivalent S	[O:1]=[S;D2;+0:2]-[#6:3]>>[*:1]=[*+1:2]-[*:3]
// Note that the next one doesn't work propertly because repeated appplications
// don't carry the cations from the previous rounds through. This should be
// fixed by implementing single-molecule transformations, but that's a longer-term
// project
//Alkaline oxide to ions	[Li,Na,K;+0:1]-[O+0:2]>>([*+1:1].[O-:2])
Bad amide tautomer1	[C:1]([OH1;D1:2])=;!@[NH1:3]>>[C:1](=[OH0:2])-[NH2:3]
Bad amide tautomer2	[C:1]([OH1;D1:2])=;!@[NH0:3]>>[C:1](=[OH0:2])-[NH1:3]
Halogen with no neighbors	[F,Cl,Br,I;X0;+0:1]>>[*-1:1]
Odd pyridine/pyridazine oxide structure	[C,N;-;D2,D3:1]-[N+2;D3:2]-[O-;D1:3]>>[*-0:1]=[*+1:2]-[*-:3]
Odd azide	[*:1][N-:2][N+:3]#[N:4]>>[*:1][N+0:2]=[N+:3]=[N-:4]
"""
_normalizer_params = rdMolStandardize.CleanupParameters()
_normalizer = rdMolStandardize.NormalizerFromData(
    _normalization_transforms, _normalizer_params
)


def flatten_tartrate_mol(m: Chem.Mol) -> Chem.Mol:
    tartrate = Chem.MolFromSmarts("OC(=O)C(O)C(O)C(=O)O")
    # make sure we only match free tartrate/tartaric acid fragments
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.adjustDegree = True
    params.adjustDegreeFlags = Chem.AdjustQueryWhichFlags.ADJUST_IGNORENONE
    tartrate = Chem.AdjustQueryProperties(tartrate, params)
    matches = m.GetSubstructMatches(tartrate)
    if matches:
        m = Chem.Mol(m)
        for match in matches:
            m.GetAtomWithIdx(match[3]).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            m.GetAtomWithIdx(match[5]).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return m


def md_minimize_energy(mol: Chem.Mol) -> Tuple[Chem.Mol, bool]:
    """
    Run Constrained Energy minimisation with MMF94
    :param mol: input conformer
    :return: optimised conformer
    """
    # Prepare the MMFF properties and force field
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    forcefield = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=0)

    # Add position constraints to every heavy atom with a moderately large force constant
    for atom in mol.GetAtoms():
        forcefield.MMFFAddPositionConstraint(atom.GetIdx(), 0.2, 800.0)

    # Initialize and minimize with limited steps
    forcefield.Initialize()
    res = forcefield.Minimize(maxIts=1000, energyTol=1e-08)

    return mol, res


def standardize_mol(mol: Chem.Mol, optimize_geometry: bool = True) -> Chem.Mol:
    """
    Molecule Standardization
    :param mol: input conformer
    :param optimize_geometry: if MMFF94 optimisation is required
    :return: standardized conformer
    """
    try:
        # Leave only largest fragment
        m = rdMolStandardize.FragmentParent(mol)
        # Kekulize
        Chem.Kekulize(m)
        # Flatten Tartrates
        m = flatten_tartrate_mol(m)

        # Sanitise
        Chem.SanitizeMol(m)

        if optimize_geometry:
            m = Chem.AddHs(m, addCoords=True)
            std_mol, _ = md_minimize_energy(m)
            std_mol = Chem.RemoveHs(std_mol)
        else:
            std_mol = m

    except:
        std_mol = None

    return std_mol
