import pickle
import time

from openbabel import openbabel
from rdkit import Chem

from src.mlconfgen import (
    MLConformerGenerator,
    evaluate_samples,
)

from src.mlconfgen.utils import standardize_mol, redefine_bonds, prepare_adj_mat_seer_input

device = "cpu"
generator = MLConformerGenerator(device=device)
# source_path = "./data/full_15_39_atoms_conf_chembl.inchi"
n_samples = 100
max_variance = 2
# mode = "STRUCTURE SEER"  # or "OPENBABEL"

# Configure OpenBabel Conversion

ob_conversion = openbabel.OBConversion()
# Tell Open Babel we’ll be reading an XYZ file
ob_conversion.SetInAndOutFormats("xyz", "mol")


references = Chem.SDMolSupplier("./1000_ccdc_validation_set.sdf")
n_ref = len(references)
expected_n_samples = n_samples * n_ref


# def exact_match(mol, source):
#     whs_mol = Chem.RemoveHs(mol)
#     s_mol = Chem.MolFromSmiles(Chem.MolToSmiles(whs_mol))
#     sample_inchi = Chem.MolToInchi(s_mol)
#
#     with open(source, "r") as f:
#         lines = f.readlines()
#         for line in lines:
#             cid, inchi = line.replace("\n", "").split("\t")
#             if sample_inchi == inchi:
#                 return True
#
#     return False


def get_samples(name: str):
    with open(f"./raw_samples/{name}_100_samples.pkl", "rb") as f:
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

    # for i, adj_mat in enumerate(adj_mat_batch):
    #     f_mol = redefine_bonds(canonicalised_samples[i], adj_mat)
    #     std_mol = standardize_mol(mol=f_mol, optimize_geometry=True)
    #     if std_mol:
    #         optimised_conformers.append(std_mol)

    return optimised_conformers


def predict_bods_openbabel(raw_samples, ob_conversion):
    optimised_conformers = []
    for sample in raw_samples:
        xyz_string = Chem.MolToXYZBlock(sample)
        # Create an empty OBMol to hold our molecule
        ob_mol = openbabel.OBMol()
        ob_conversion.ReadString(ob_mol, xyz_string)
        ob_mol.ConnectTheDots()
        ob_mol.PerceiveBondOrders()

        mol_block_str = ob_conversion.WriteString(ob_mol)

        rdkit_mol = Chem.MolFromMolBlock(mol_block_str)

        # std_mol = standardize_mol(mol=rdkit_mol, optimize_geometry=True)
        # if std_mol:
        #     optimised_conformers.append(std_mol)

    return optimised_conformers


structure_seer_time = 0
# structure_seer_n_samples = 0
openbabel_time = 0
# openbabel_n_samples = 0

for i, reference in enumerate(references):
    print(f"Analysing samples for reference compound {i + 1} of {n_ref}")

    # Get a name of the reference
    ref_name = reference.GetProp("_Name")
    reference = Chem.RemoveHs(reference)
    ref_n_atoms = reference.GetNumAtoms()

    raw_samples = get_samples(ref_name)

    print(f" Predicting bonds with STRUCTURE SEER")

    struct_seer_start = time.time()
    struct_seer_samples = predict_bods_gcn(raw_samples, generator)
    struct_seer_finish = time.time() - struct_seer_start

    print(f"Finished in {struct_seer_finish}")

    structure_seer_time += struct_seer_finish
    # structure_seer_n_samples += len(struct_seer_samples)

    print(f" Predicting bonds with OPENBABEL")

    obabel_start = time.time()
    obabel_samples = predict_bods_openbabel(raw_samples, ob_conversion)
    obabel_finish = time.time() - obabel_start

    print(f"Finished in {obabel_finish}")

    # openbabel_n_samples += len(obabel_samples)
    openbabel_time += obabel_finish


with open("../struct_seer_obabel_speed_report.txt", "w+") as f:
    f.write(f"SPEED COMPARISON REPORT\n\n")
    f.write(f"STRUCTURE SEER Total Time {structure_seer_time} sec\n")
    # f.write(f"STRUCTURE SEER Total valid samples {structure_seer_n_samples}\n\n")
    f.write(f"OPENBABEL Total Time {openbabel_time} sec\n")
    # f.write(f"OPENBABEL Total valid samples {openbabel_n_samples}\n")


