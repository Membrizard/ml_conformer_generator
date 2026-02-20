from rdkit import Chem
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

uncharger = rdMolStandardize.Uncharger()  # default behavior


def neutralize_mol(mol) -> str:
    mol = uncharger.uncharge(mol)
    return mol


def fragment_to_query_neutralized(frag_mol: Chem.Mol) -> Chem.Mol:
    frag_mol = neutralize_mol(frag_mol)   # or just uncharger.uncharge(frag_mol)
    return Chem.MolToSmarts(frag_mol)


f_suppl = Chem.SDMolSupplier("./data/ifm_evaluation/fixed_fragment/evaluation_ifm_fragments.sdf")

counter = 0
for mol in tqdm(f_suppl):
    smarts = fragment_to_query_neutralized(mol)
    print(smarts)

# supplier = Chem.SDMolSupplier("./data/ifm_evaluation/fixed_fragment/evaluation_ifm_fixed_fragment_generation.sdf")
#
# count = 0
# average = 0
#
# average_ss_score = 0
#
# average_dict = {}
# count_dict = {}
# for mol in tqdm(supplier):
#     count += 1
#     shape_sim = float(mol.GetProp("shape_similarity"))
#     ref_n_atoms = int(mol.GetProp("ref_n_atoms"))
#     n_valid_samples = int(mol.GetProp("valid_samples_in_generation")) / 100
#     average_ss_score += shape_sim
#
#     average += n_valid_samples
#     if ref_n_atoms in average_dict.keys():
#         count_dict[ref_n_atoms] += 1
#         average_dict[ref_n_atoms] += shape_sim
#     else:
#         count_dict[ref_n_atoms] = 1
#         average_dict[ref_n_atoms] = shape_sim
#
# print(f"Average valid samples: {round(average / count, 4)}")
# print(f"Average shape similarity: {round(average_ss_score / count, 4)}")
# for key in average_dict:
#     print(f"{key}: {round(average_dict[key] / count_dict[key], 4)}")