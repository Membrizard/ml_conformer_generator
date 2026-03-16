from rdkit import Chem
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

uncharger = rdMolStandardize.Uncharger()  # default behavior


def fragment_to_query_neutralized(frag_mol: Chem.Mol) -> str:
    return Chem.MolToSmarts(uncharger.uncharge(frag_mol))


fragment_supplier = Chem.SDMolSupplier("./data/ifm_evaluation/fixed_fragment/evaluation_ifm_fragments.sdf")
supplier = Chem.SDMolSupplier("./data/ifm_evaluation/fixed_fragment/evaluation_ifm_fixed_fragment_generation.sdf")

match_counter = 0
counter = 0
for gen_mol in tqdm(supplier):
    counter += 1
    frag_mol = Chem.MolFromSmiles(gen_mol.GetProp("fixed_fragment_smiles"))
    prepared_q = Chem.MolFromSmarts(fragment_to_query_neutralized(frag_mol))
    has_match = gen_mol.HasSubstructMatch(prepared_q)
    if has_match:
        match_counter += 1

print(f"Total mols: {counter} mols with matches: {match_counter}")
print(f"Success fraction: {round(match_counter / counter, 2)}")
