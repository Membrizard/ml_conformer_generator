#
#  Based on ChEMBL_StructurePipeline project
#  Copyright (c) 2019 Greg Landrum
#  All rights reserved.
#
#  This file is based on a part of the ChEMBL_StructurePipeline project.
#  The contents are covered by the terms of the MIT license
#  which is included in the file LICENSE, found at the root
#  of the source tree.


import os
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import AllChem


def kekulize_mol(m):
    Chem.Kekulize(m)
    return m


def update_mol_valences(m):
    m = Chem.Mol(m)
    m.UpdatePropertyCache(strict=False)
    return m


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

_alkoxide_pattern = Chem.MolFromSmarts("[Li,Na,K;+0]-[#7,#8;+0]")


def normalize_mol(m):
    """ """
    Chem.FastFindRings(m)
    if m.HasSubstructMatch(_alkoxide_pattern):
        m = Chem.RWMol(m)
        for match in m.GetSubstructMatches(_alkoxide_pattern):
            m.RemoveBond(match[0], match[1])
            m.GetAtomWithIdx(match[0]).SetFormalCharge(1)
            m.GetAtomWithIdx(match[1]).SetFormalCharge(-1)
    res = _normalizer.normalize(m)
    return res


def remove_hs_from_mol(m):
    """removes most Hs

    Hs that are preserved by the RDKit's Chem.RemoveHs() will not
    be removed.

    Additional exceptions:
    - Hs with a wedged/dashed bond to them
    - Hs bonded to atoms with tetrahedral stereochemistry set
    - Hs bonded to atoms that have three (or more) ring bonds that are not simply protonated
    - Hs bonded to atoms in a non-default valence state that are not simply protonated


    For the above, the definition of "simply protonated" is an atom with charge = +1 and
    a valence that is one higher than the default.

    """
    # we need ring info, so be sure it's there (this won't do anything if the rings
    # have already been found)
    Chem.FastFindRings(m)
    if m.NeedsUpdatePropertyCache():
        m.UpdatePropertyCache(strict=False)
    SENTINEL = 100
    for atom in m.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetDegree() == 1 and not atom.GetIsotope():
            nbr = atom.GetNeighbors()[0]
            bnd = atom.GetBonds()[0]
            preserve = False
            if bnd.GetBondDir() in (
                Chem.BondDir.BEGINWEDGE,
                Chem.BondDir.BEGINDASH,
            ) or (
                bnd.HasProp("_MolFileBondStereo")
                and bnd.GetUnsignedProp("_MolFileBondStereo") in (1, 6)
            ):
                preserve = True
            else:
                is_protonated = (
                    nbr.GetFormalCharge() == 1
                    and nbr.GetExplicitValence()
                    == Chem.GetPeriodicTable().GetDefaultValence(nbr.GetAtomicNum()) + 1
                )
                if nbr.GetChiralTag() in (
                    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
                    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                ):
                    preserve = True
                elif not is_protonated:
                    if (
                        nbr.GetExplicitValence()
                        > Chem.GetPeriodicTable().GetDefaultValence(nbr.GetAtomicNum())
                    ):
                        preserve = True
                    else:
                        ringBonds = [
                            b
                            for b in nbr.GetBonds()
                            if m.GetRingInfo().NumBondRings(b.GetIdx())
                        ]
                        if len(ringBonds) >= 3:
                            preserve = True
            if preserve:
                # we're safe picking an arbitrary high value since you can't do this in a mol block:
                atom.SetIsotope(SENTINEL)

    res = Chem.RemoveHs(m, sanitize=False)
    for atom in res.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetIsotope() == SENTINEL:
            atom.SetIsotope(0)
    return res


# def remove_sgroups_from_mol(m):
#     """removes all Sgroups"""
#     Chem.ClearMolSubstanceGroups(m)
#     return m


def uncharge_mol(m):
    """

    >>> def uncharge_smiles(smi): return Chem.MolToSmiles(uncharge_mol(Chem.MolFromSmiles(smi)))
    >>> uncharge_smiles('[NH3+]CCC')
    'CCCN'
    >>> uncharge_smiles('[NH3+]CCC[O-]')
    'NCCCO'
    >>> uncharge_smiles('C[N+](C)(C)CCC[O-]')
    'C[N+](C)(C)CCC[O-]'
    >>> uncharge_smiles('CC[NH+](C)C.[Cl-]')
    'CCN(C)C.Cl'
    >>> uncharge_smiles('CC(=O)[O-]')
    'CC(=O)O'
    >>> uncharge_smiles('CC(=O)[O-].[Na+]')
    'CC(=O)[O-].[Na+]'
    >>> uncharge_smiles('[NH3+]CC(=O)[O-].[Na+]')
    'NCC(=O)[O-].[Na+]'
    >>> uncharge_smiles('CC(=O)[O-].C[NH+](C)C')
    'CC(=O)O.CN(C)C'

    Alcohols are protonated before acids:

    >>> uncharge_smiles('[O-]C([N+](C)C)CC(=O)[O-]')
    'C[N+](C)C(O)CC(=O)[O-]'

    And the neutralization is done in a canonical order, so atom ordering of the input
    structure isn't important:

    >>> uncharge_smiles('C[N+](C)(C)CC([O-])CC[O-]')
    'C[N+](C)(C)CC([O-])CCO'
    >>> uncharge_smiles('C[N+](C)(C)CC(CC[O-])[O-]')
    'C[N+](C)(C)CC([O-])CCO'

    """
    uncharger = rdMolStandardize.Uncharger(canonicalOrder=True)
    res = uncharger.uncharge(m)
    res.UpdatePropertyCache(strict=False)
    return res


def flatten_tartrate_mol(m):
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


_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_solvents_file = os.path.join(_data_dir, "solvents.smi")
_salts_file = os.path.join(_data_dir, "salts.smi")


def md_minimize_energy(mol):
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


def standardize_mol(mol, optimize_geometry: bool = True):
    try:
        # Leave only largest fragment
        m = rdMolStandardize.FragmentParent(mol)
        # Kekulize
        m = kekulize_mol(m)
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
