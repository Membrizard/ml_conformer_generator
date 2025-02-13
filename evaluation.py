from ml_conformer_generator import MLConformerGenerator
from cheminformatics import evaluate_samples
from rdkit import Chem


def exact_match(mol, source):
    sample_inchi = Chem.MolToInchi(mol)

    with open(source, "r") as f:
        lines = f.readlines()
        for line in lines:
            cid, inchi = line.replace("\n", "").split('\t')
            if sample_inchi == inchi:
                return True

    return False


device = "cuda"
generator = MLConformerGenerator(device=device)
source_path = "./data/full_15_39_atoms_conf_chembl.inchi"
n_samples = 20
max_variance = 5

references = Chem.SDMolSupplier("./data/100_ccdc_validation_set.sdf")
n_ref = len(references)
expected_n_samples = n_samples * n_ref

node_dist_dict = dict()  # n_atoms : number of samples generated for mols with the given n_atoms
variance_dist_dict = dict()  # n_atom variance from ref n_atoms : number of samples with such variance

# variance_context_errors = dict()  # n_atom variance from ref n_atoms : error in context
# ref_mol_size_context_errors = dict()  # n_atoms: error in context

variance_shape_tanimoto_scores = dict()  # n_atom variance from ref n_atoms : shape tanimoto
ref_mol_size_shape_tanimoto_scores = dict()  # n_atoms: shape tanimoto

variance_chem_tanimoto_scores = dict()  # n_atom variance from ref n_atoms : chemical tanimoto
ref_mol_size_chem_tanimoto_scores = dict()  # n_atoms: chemical tanimoto

ref_mol_size_valid = dict()  # n_atoms: number of valid molecules generated in % of requested to be generated

chem_unique_samples = 0  # number of chemically unique samples generated
valid_samples = 0  # number of valid samples generated

for i, reference in enumerate(references):
    print(f"Analysing samples for reference compound {i + 1} of {n_ref}")
    reference = Chem.RemoveHs(reference)
    ref_n_atoms = reference.GetNumAtoms()

    samples = generator.generate_conformers(reference_conformer=reference, n_samples=n_samples, variance=max_variance)
    _, std_samples = evaluate_samples(reference, samples)

    valid_samples += len(std_samples)

    # Log fraction of valid molecules
    if ref_n_atoms in node_dist_dict.keys():
        ref_mol_size_valid[ref_n_atoms] += len(std_samples) / n_samples
    else:
        ref_mol_size_valid[ref_n_atoms] = len(std_samples) / n_samples

    for std_sample in std_samples:

        sample_mol = Chem.MolFromMolBlock(std_sample['mol_block'])

        # Check for sample uniqueness

        if exact_match(sample_mol, source_path):
            chem_unique_samples += 1

        sample_num_atoms = sample_mol.GetNumAtoms()
        variance = ref_n_atoms - sample_num_atoms  # -> Can be negative intentionally
        # sample_conformer = sample_mol.GetConformer()
        # sample_coord = torch.tensor(sample_conformer.GetPositions(), dtype=torch.float32)
        #
        # sample_context, _ = get_context_shape(sample_coord)
        # context_error = torch.abs(reference_context - sample_context)

        # Log the error in context generation and tanimoto scores for ref_n_atoms
        if ref_n_atoms in node_dist_dict.keys():
            node_dist_dict[ref_n_atoms] += 1
            # ref_mol_size_context_errors[ref_n_atoms] += context_error
            ref_mol_size_shape_tanimoto_scores[ref_n_atoms] += std_sample['shape_tanimoto']
            ref_mol_size_chem_tanimoto_scores[ref_n_atoms] += std_sample['chemical_tanimoto']

        else:
            node_dist_dict[ref_n_atoms] = 1
            # ref_mol_size_context_errors[ref_n_atoms] = context_error
            ref_mol_size_shape_tanimoto_scores[ref_n_atoms] = std_sample['shape_tanimoto']
            ref_mol_size_chem_tanimoto_scores[ref_n_atoms] = std_sample['chemical_tanimoto']

        # Log the error in context generation and tanimoto scores for variance
        if variance in variance_dist_dict.keys():
            variance_dist_dict[variance] += 1
            # variance_context_errors[variance] += context_error
            variance_shape_tanimoto_scores[variance] += std_sample['shape_tanimoto']
            variance_chem_tanimoto_scores[variance] += std_sample['chemical_tanimoto']
        else:
            variance_dist_dict[variance] = 1
            # variance_context_errors[variance] = context_error
            variance_shape_tanimoto_scores[variance] = std_sample['shape_tanimoto']
            variance_chem_tanimoto_scores[variance] = std_sample['chemical_tanimoto']

valid_samples_rate = valid_samples / expected_n_samples
chem_unique_samples_rate = chem_unique_samples / valid_samples

# Calculate mean Error and Tanimoto score values, for ref_mol_size and variance

for key in node_dist_dict.keys():
    # ref_mol_size_context_errors[key] = ref_mol_size_context_errors[key] / node_dist_dict[key]
    ref_mol_size_shape_tanimoto_scores[key] = ref_mol_size_shape_tanimoto_scores[key] / node_dist_dict[key]
    ref_mol_size_chem_tanimoto_scores[key] = ref_mol_size_chem_tanimoto_scores[key] / node_dist_dict[key]

for key in variance_dist_dict.keys():
    # variance_context_errors[key] = variance_context_errors[key] / variance_dist_dict[key]
    variance_shape_tanimoto_scores[key] = variance_shape_tanimoto_scores[key] / variance_dist_dict[key]
    variance_chem_tanimoto_scores[key] = variance_chem_tanimoto_scores[key] / variance_dist_dict[key]

with open("generation_performance_report.txt", "w+") as f:
    f.write(f"Number of Contexts used for generation - {n_ref}\n")
    f.write(f"Number of Samples per Context - {n_samples}\n\n")
    f.write(
        f"Total valid molecules generated - {valid_samples} ({round(valid_samples_rate, 4) * 100}% out of requested)\n")
    f.write(
        f"From them, Chemically Unique in reference to training Dataset - {round(chem_unique_samples_rate, 4) * 100}%\n")

    # f.write("\nErrors in Context of Generated Molecules vs number of atoms in reference:\n\n")
    # for key in sorted(ref_mol_size_context_errors.keys()):
    #     f.write(f"{key}:  {ref_mol_size_context_errors[key]}\n")

    f.write("\nShape Tanimoto Scores of Generated Molecules vs number of atoms in reference:\n\n")
    for key in sorted(ref_mol_size_shape_tanimoto_scores.keys()):
        f.write(f"{key}:  {ref_mol_size_shape_tanimoto_scores[key]}\n")

    f.write("\nChemical Tanimoto Scores of Generated Molecules vs number of atoms in reference:\n\n")
    for key in sorted(ref_mol_size_chem_tanimoto_scores.keys()):
        f.write(f"{key}:  {ref_mol_size_chem_tanimoto_scores[key]}\n")

    # f.write("\nErrors in Context of Generated Molecules vs variation of number of atoms from reference:\n\n")
    # for key in sorted(variance_context_errors.keys()):
    #     f.write(f"{key}:  {variance_context_errors[key]}\n")

    f.write("\nShape Tanimoto Scores of Generated Molecules vs variation of number of atoms from reference:\n\n")
    for key in sorted(variance_shape_tanimoto_scores.keys()):
        f.write(f"{key}:  {variance_shape_tanimoto_scores[key]}\n")

    f.write("\nChemical Tanimoto Scores of Generated Molecules vs variation of number of atoms from reference:\n\n")
    for key in sorted(variance_chem_tanimoto_scores.keys()):
        f.write(f"{key}:  {variance_chem_tanimoto_scores[key]}\n")

# - What is the error in context of generated samples vs number of atoms in the reference and variance
# - What is the average shape tanimoto similarity of generated samples vs number of atoms in the reference and variance
# - How many valid samples (after cheminformatics pipeline) was generated (as % from generated with edm) vs number of atoms in the reference
# - How many chemically unique samples (which the model has never seen) was generated in total (as % of the number of all valid