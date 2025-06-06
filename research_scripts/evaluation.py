import os
import time

from rdkit import Chem

from mlconfgen import (
    MLConformerGenerator,
    evaluate_samples,
)


def exact_match(taget, source):
    reader = Chem.SDMolSupplier(taget)

    source_inchi = []
    with open(source, "r") as f:
        lines = f.readlines()
        for line in lines:
            _, inchi = line.replace("\n", "").split("\t")

            source_inchi.append(inchi)

    source_inchi = set(source_inchi)

    taget_inchi = []
    counter = 0
    for mol in reader:
        whs_mol = Chem.RemoveHs(mol)
        s_mol = Chem.MolFromSmiles(Chem.MolToSmiles(whs_mol))
        sample_inchi = Chem.MolToInchi(s_mol)

        taget_inchi.append(sample_inchi)
        counter += 1

    target_inchi = set(taget_inchi)
    n_unique_target = len(target_inchi)

    n_intersection = len(source_inchi.intersection(target_inchi))

    set_unique = counter - n_intersection

    return {"unique_within_batch": n_unique_target, "unique_train_set": set_unique}


device = "cuda"
source_path = "./full_15_39_atoms_conf_chembl.inchi"
n_samples = 100
max_variance = 2

references = Chem.SDMolSupplier("./data/1000_ccdc_validation_set.sdf")
n_ref = len(references)
expected_n_samples = n_samples * n_ref

n_dif_steps = [100]
folder = "./time_steps_experiments"
os.makedirs(folder, exist_ok=True)

for time_steps in n_dif_steps:
    generator = MLConformerGenerator(device=device, diffusion_steps=time_steps)
    out_file_name = f"{folder}/{time_steps}_time_steps_generated_samples.sdf"
    writer = Chem.SDWriter(out_file_name)

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

    # chem_unique_samples = 0  # number of chemically unique samples generated
    valid_samples = 0  # number of valid samples generated
    average_shape_tanimoto = 0
    average_chemical_tanimoto = 0
    total_gen_time = 0
    max_shape_tanimoto = dict()

    for i, reference in enumerate(references):
        print(
            f"Analysing samples for reference compound {i + 1} of {n_ref} number of timesteps - {time_steps}"
        )
        reference = Chem.RemoveHs(reference)
        ref_n_atoms = reference.GetNumAtoms()

        gen_start = time.time()
        samples = generator.generate_conformers(
            reference_conformer=reference, n_samples=n_samples, variance=max_variance
        )
        gen_time = round(time.time() - gen_start, 2)
        total_gen_time += gen_time

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
            # Add a sample to gen samples file
            writer.write(sample_mol)

            sample_num_atoms = sample_mol.GetNumAtoms()
            variance = (
                ref_n_atoms - sample_num_atoms
            )  # -> Can be negative intentionally

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
                variance_chem_tanimoto_scores[variance] += std_sample[
                    "chemical_tanimoto"
                ]
            else:
                variance_dist_dict[variance] = 1
                variance_shape_tanimoto_scores[variance] = std_sample["shape_tanimoto"]
                variance_chem_tanimoto_scores[variance] = std_sample[
                    "chemical_tanimoto"
                ]

    valid_samples_rate = valid_samples / expected_n_samples

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

    # Check Unique Samples
    print("Evaluating samples uniqueness...")
    unique_dict = exact_match(taget=out_file_name, source=source_path)

    chem_unique_samples_rate = unique_dict["unique_train_set"] / valid_samples
    gen_unique_samples_rate = unique_dict["unique_within_batch"] / valid_samples

    print("Done!")

    file_name = f"{folder}/{time_steps}_time_steps_generation_performance_report_1000_ref_100_samples_var_2.txt"

    with open(file_name, "w+") as f:
        f.write(f"Number of diffusion steps {time_steps}\n")

        f.write(f"Number of Contexts used for generation - {n_ref}\n")
        f.write(f"Number of Samples per Context - {n_samples}\n\n")
        f.write(f"Total time for generation - {round(total_gen_time, 2)} sec\n")
        f.write(
            f"Averaged time for generation (per reference) - {round(total_gen_time / n_ref ,2)} sec per request\n"
        )
        f.write(
            f"Averaged generation speed (per expected molecule) - {round(expected_n_samples / total_gen_time, 2)} molecule/sec\n"
        )
        f.write(
            f"Averaged generation speed (per valid molecule) - {round(valid_samples / total_gen_time ,2)} molecule/sec\n"
        )

        f.write(
            f"Total valid molecules generated - {valid_samples} ({round(valid_samples_rate, 4) * 100}% out of requested)\n"
        )
        f.write(
            f"From them, Chemically Unique in reference to training Dataset - {round(chem_unique_samples_rate, 4) * 100}%\n"
        )

        f.write(
            f"From them, Chemically Unique withing the Generated Set - {round(gen_unique_samples_rate, 4) * 100}%\n"
        )

        f.write(
            f"Average Shape Tanimoto Similarity - {round(average_shape_tanimoto / valid_samples, 4) * 100}%\n"
        )
        f.write(
            f"Average Chemical Tanimoto Similarity - {round(average_chemical_tanimoto / valid_samples, 4) * 100}%\n"
        )

        f.write(
            "\nAverage Shape Tanimoto Scores of Generated Molecules vs number of atoms in reference:\n\n"
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
