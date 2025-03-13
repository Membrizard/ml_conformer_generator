from openbabel import openbabel
import time

from ml_conformer_generator.ml_conformer_generator import (
    MLConformerGenerator,
    evaluate_samples,
)
from rdkit import Chem
import pickle


def exact_match(mol, source):
    whs_mol = Chem.RemoveHs(mol)
    s_mol = Chem.MolFromSmiles(Chem.MolToSmiles(whs_mol))
    sample_inchi = Chem.MolToInchi(s_mol)

    with open(source, "r") as f:
        lines = f.readlines()
        for line in lines:
            cid, inchi = line.replace("\n", "").split("\t")
            if sample_inchi == inchi:
                return True

    return False


def get_samples(name: str):
    with open(f"./raw_samples/{name}_100_samples.pkl", 'rb') as f:
        raw_samples = pickle.load(f)

    return raw_samples


def predict_bods_gcn(raw_samples, generator):
    (
        el_batch,
        dm_batch,
        b_adj_mat_batch,
        canonicalised_samples,
    ) = prepare_adj_mat_seer_input(
        mols=raw_samples,
        n_samples=n_samples,
        dimension=generator.dimension,
        device=generator.device,
    )

    adj_mat_batch = generator.adj_mat_seer(
        elements=el_batch, dist_mat=dm_batch, adj_mat=b_adj_mat_batch
    )

    adj_mat_batch = adj_mat_batch.to("cpu")

    # Append generated bonds and standardise existing samples
    optimised_conformers = []

    for i, adj_mat in enumerate(adj_mat_batch):
        f_mol = redefine_bonds(canonicalised_samples[i], adj_mat)
        std_mol = standardize_mol(mol=f_mol, optimize_geometry=True)
        if std_mol:
            optimised_conformers.append(std_mol)

    return optimised_conformers


def predict_bods_openbabel(raw_samples, obconversion):
    optimised_conformers = []
    for sample in raw_samples:
        xyz_string = Chem.MolToXYZBlock(sample)
        # Create an empty OBMol to hold our molecule
        ob_mol = openbabel.OBMol()
        obconversion.ReadString(ob_mol, xyz_string)
        ob_mol.ConnectTheDots()
        ob_mol.PerceiveBondOrders()

        mol_block_str = obconversion.WriteString(ob_mol)

        rdkit_mol = Chem.MolFromMolBlock(mol_block_str)

        std_mol = standardize_mol(mol=rdkit_mol, optimize_geometry=True)
        if std_mol:
            optimised_conformers.append(std_mol)

    return optimised_conformers




device = "cpu"
generator = MLConformerGenerator(device=device)
source_path = "./data/full_15_39_atoms_conf_chembl.inchi"
n_samples = 100
max_variance = 2
mode = "STRUCTURE SEER"  # or "OPENBABEL"

# Configure OpenBabel Conversion

obConversion = openbabel.OBConversion()
# Tell Open Babel we’ll be reading an XYZ file
obConversion.SetInAndOutFormats("xyz", "mol")


references = Chem.SDMolSupplier("./data/1000_ccdc_validation_set.sdf")
n_ref = len(references)
expected_n_samples = n_samples * n_ref

node_dist_dict = (
    dict()
)  # n_atoms : number of samples generated for mols with the given n_atoms
variance_dist_dict = (
    dict()
)  # n_atom variance from ref n_atoms : number of samples with such variance

variance_shape_tanimoto_scores = (
    dict()
)  # n_atom variance from ref n_atoms : shape tanimoto
ref_mol_size_shape_tanimoto_scores = dict()  # n_atoms: shape tanimoto

variance_chem_tanimoto_scores = (
    dict()
)  # n_atom variance from ref n_atoms : chemical tanimoto
ref_mol_size_chem_tanimoto_scores = dict()  # n_atoms: chemical tanimoto

ref_mol_size_valid = (
    dict()
)  # n_atoms: number of valid molecules generated in % of requested to be generated

chem_unique_samples = 0  # number of chemically unique samples generated
valid_samples = 0  # number of valid samples generated
average_shape_tanimoto = 0
average_chemical_tanimoto = 0
max_shape_tanimoto = dict()

for i, reference in enumerate(references):
    print(f"Analysing samples for reference compound {i + 1} of {n_ref}")

    # Get a name of the reference
    ref_name = reference.GetProp("_Name")
    reference = Chem.RemoveHs(reference)
    ref_n_atoms = reference.GetNumAtoms()

    samples = get_samples(ref_name)








    start = time.time()
    _, std_samples = evaluate_samples(reference, samples)

    n_std_samples = len(std_samples)
    print(
        f"Tanimoto similarity for {n_std_samples} calculated in {time.time() - start} sec"
    )
    valid_samples += n_std_samples
    print(f" {n_std_samples} valid samples out of {n_samples} requested")
    # Log fraction of valid molecules
    if ref_n_atoms in node_dist_dict.keys():
        ref_mol_size_valid[ref_n_atoms] += n_std_samples / n_samples
    else:
        ref_mol_size_valid[ref_n_atoms] = n_std_samples / n_samples

    for std_sample in std_samples:
        sample_mol = Chem.MolFromMolBlock(std_sample["mol_block"], removeHs=True)

        # Check for sample uniqueness
        match = exact_match(sample_mol, source_path)
        if not match:
            chem_unique_samples += 1

        sample_num_atoms = sample_mol.GetNumAtoms()
        variance = ref_n_atoms - sample_num_atoms  # -> Can be negative intentionally

        # Log the error in context generation and tanimoto scores for ref_n_atoms
        if ref_n_atoms in node_dist_dict.keys():
            node_dist_dict[ref_n_atoms] += 1
            ref_mol_size_shape_tanimoto_scores[ref_n_atoms] += std_sample[
                "shape_tanimoto"
            ]
            ref_mol_size_chem_tanimoto_scores[ref_n_atoms] += std_sample[
                "chemical_tanimoto"
            ]
            average_shape_tanimoto += std_sample["shape_tanimoto"]

            average_chemical_tanimoto += std_sample["chemical_tanimoto"]
            if std_sample["shape_tanimoto"] >= max_shape_tanimoto[ref_n_atoms]:
                max_shape_tanimoto[ref_n_atoms] = std_sample["shape_tanimoto"]

        else:
            node_dist_dict[ref_n_atoms] = 1
            ref_mol_size_shape_tanimoto_scores[ref_n_atoms] = std_sample[
                "shape_tanimoto"
            ]
            ref_mol_size_chem_tanimoto_scores[ref_n_atoms] = std_sample[
                "chemical_tanimoto"
            ]
            max_shape_tanimoto[ref_n_atoms] = std_sample["shape_tanimoto"]

        # Log the error in context generation and tanimoto scores for variance
        if variance in variance_dist_dict.keys():
            variance_dist_dict[variance] += 1
            variance_shape_tanimoto_scores[variance] += std_sample["shape_tanimoto"]
            variance_chem_tanimoto_scores[variance] += std_sample["chemical_tanimoto"]
        else:
            variance_dist_dict[variance] = 1
            variance_shape_tanimoto_scores[variance] = std_sample["shape_tanimoto"]
            variance_chem_tanimoto_scores[variance] = std_sample["chemical_tanimoto"]

valid_samples_rate = valid_samples / expected_n_samples
chem_unique_samples_rate = chem_unique_samples / valid_samples

# Calculate mean Error and Tanimoto score values, for ref_mol_size and variance

for key in node_dist_dict.keys():
    ref_mol_size_shape_tanimoto_scores[key] = (
        ref_mol_size_shape_tanimoto_scores[key] / node_dist_dict[key]
    )
    ref_mol_size_chem_tanimoto_scores[key] = (
        ref_mol_size_chem_tanimoto_scores[key] / node_dist_dict[key]
    )

for key in variance_dist_dict.keys():
    variance_shape_tanimoto_scores[key] = (
        variance_shape_tanimoto_scores[key] / variance_dist_dict[key]
    )
    variance_chem_tanimoto_scores[key] = (
        variance_chem_tanimoto_scores[key] / variance_dist_dict[key]
    )

with open("generation_performance_report.txt", "w+") as f:
    f.write(f"GENERATION PERFORMANCE REPORT")
    f.write(f"Bonds are predicted using {mode}")
    f.write(f"Number of Contexts used for generation - {n_ref}\n")
    f.write(f"Number of Samples per Context - {n_samples}\n\n")
    f.write(
        f"Total valid molecules generated - {valid_samples} ({round(valid_samples_rate, 4) * 100}% out of requested)\n"
    )
    f.write(
        f"From them, Chemically Unique in reference to training Dataset - {round(chem_unique_samples_rate, 4) * 100}%\n"
    )

    f.write(
        f"Average Shape Tanimoto Similarity - {round(average_shape_tanimoto / valid_samples, 4) * 100}%\n"
    )
    f.write(
        f"Average Chemical Tanimoto Similarity - {round(average_chemical_tanimoto / valid_samples, 4) * 100}%\n"
    )

    f.write(
        "\n Average Shape Tanimoto Scores of Generated Molecules vs number of atoms in reference:\n\n"
    )
    for key in sorted(ref_mol_size_shape_tanimoto_scores.keys()):
        f.write(f"{key}:  {ref_mol_size_shape_tanimoto_scores[key]}\n")

    f.write(
        "\n Maximal Shape Tanimoto Scores of Generated Molecules vs number of atoms in reference:\n\n"
    )
    for key in sorted(max_shape_tanimoto.keys()):
        f.write(f"{key}:  {max_shape_tanimoto[key]}\n")

    f.write(
        "\nChemical Tanimoto Scores of Generated Molecules vs number of atoms in reference:\n\n"
    )
    for key in sorted(ref_mol_size_chem_tanimoto_scores.keys()):
        f.write(f"{key}:  {ref_mol_size_chem_tanimoto_scores[key]}\n")

    f.write(
        "\nNote: The abnormalities in variance of the number of atoms may be present\n "
        "due to striping of small unconnected fragments of the generated molecules by the"
        " cheminformatics standardisation pipline\n"
    )

    f.write(
        "\nShape Tanimoto Scores of Generated Molecules vs variation of number of atoms from reference:\n\n"
    )
    for key in sorted(variance_shape_tanimoto_scores.keys()):
        f.write(f"{key}:  {variance_shape_tanimoto_scores[key]}\n")

    f.write(
        "\nChemical Tanimoto Scores of Generated Molecules vs variation of number of atoms from reference:\n\n"
    )
    for key in sorted(variance_chem_tanimoto_scores.keys()):
        f.write(f"{key}:  {variance_chem_tanimoto_scores[key]}\n")





























