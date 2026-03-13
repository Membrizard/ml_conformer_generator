import logging
import time
from rdkit import Chem, RDLogger

from tqdm import tqdm


from src.mlconfgen import evaluate_samples, MLConformerGenerator
from src.mlconfgen.inertial_fragment_matching import ff_inertial_fragment_matching, predict_bonds_openbabel


# LOGGING
RDLogger.DisableLog('rdApp.*')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [IFM Evaluation log] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

# CONFIGURATION
# ---------------------------------
DATASET = "./data/ifm_fixed_fragment_test_6_20.sdf"
N_SAMPLES = 100
VARIANCE = 1
OPTIMIZE_GEOMETRY = False
DEVICE = "cuda"

GENERATOR = MLConformerGenerator(
    edm_weights="./licensed_edm_moi_chembl_6_39_final.pt",
    adj_mat_seer_weights="./adj_mat_seer_chembl_15_39.pt",
    device=DEVICE,
    diffusion_steps=100,  # Light generation for debugging
)

MERGER = MLConformerGenerator(
    edm_weights="./edm_moi_chembl_15_39.pt",
    adj_mat_seer_weights="./adj_mat_seer_chembl_15_39.pt",
    device=DEVICE,
    diffusion_steps=100,  # Light generation for debugging
)
VERBOSE = False
MIN_FRAG_SIZE = 6
MAX_FRAG_SIZE = 20
DSTEPS_MERGING = 10
RESAMPLE_STEPS = 0


OUTPUT_SDF = "./data/smoke_test_ifm_out.sdf"
FF_OUTPUT_SDF = "./data/fragments_smoke_test_ifm_out.sdf"
# ---------------------------------


def ff_generate_molecules(
                       ref_mol: Chem.Mol,
                       n_samples: int = N_SAMPLES,
                       variance: int = VARIANCE,
                       optimize_geometry: bool = OPTIMIZE_GEOMETRY,
                       generator: MLConformerGenerator = GENERATOR,
                       merger: MLConformerGenerator = MERGER,
                       min_frag_size: int = MIN_FRAG_SIZE,
                       max_frag_size: int = MAX_FRAG_SIZE,
                       diffusion_steps_merging: int = DSTEPS_MERGING,
                       resample_steps: int = RESAMPLE_STEPS,
                       blend_power: int = 3,
                       verbose: bool = VERBOSE,
                       ):

    if verbose:
        logging.info("Inertial Fragment Matching happening...")

    final_mols, fixed_fragment = ff_inertial_fragment_matching(
            ref_conformer=ref_mol,
            generator=generator,
            merger=merger,
            n_samples=n_samples,
            variance=variance,
            resample_steps=resample_steps,
            blend_power=blend_power,
            merging_diffusion_level=diffusion_steps_merging,
            min_frag_size=min_frag_size,
            max_frag_size=max_frag_size,
            max_iter=200,
            verbose=verbose,
        )
    if verbose:
        logging.info("Inertial Fragment Matching happened!")

    obabel_mols = []

    # Switched to deterministic bond prediction
    for mol in final_mols:
        f_mol = predict_bonds_openbabel(mol, optimize_geometry=optimize_geometry)
        if f_mol:
            obabel_mols.append(f_mol)

    _, std_samples = evaluate_samples(ref_mol, obabel_mols)

    return std_samples, fixed_fragment


def inpaint_strategy():
    return None


def log_samples_with_metadata(
                generated_samples: list,
                ref_mol: Chem.Mol,
                fixed_fragment: Chem.Mol,
                generation_id: str,
                writer: Chem.SDWriter,
                fragment_writer: Chem.SDWriter,
                variance: int = VARIANCE,
                requested_samples: int = N_SAMPLES,
                ) -> None:

    valid_samples = len(generated_samples)
    ref_n_atoms = ref_mol.GetNumHeavyAtoms()
    ref_name = ref_mol.GetProp("_Name")

    fixed_fragment.SetProp("_Name", generation_id)

    fragment_writer.write(fixed_fragment)

    ff_smiles = Chem.MolToSmiles(fixed_fragment)

    for i, sample in enumerate(generated_samples):
        shape_similarity_score = sample["shape_tanimoto"]
        chemical_similarity_score = sample["chemical_tanimoto"]
        gen_mol = Chem.MolFromMolBlock(sample["mol_block"])

        gen_mol.SetProp("fixed_fragment_smiles", ff_smiles)
        gen_mol.SetProp("reference_name", ref_name)
        gen_mol.SetProp("shape_similarity", str(shape_similarity_score))
        gen_mol.SetProp("chemical_similarity", str(chemical_similarity_score))
        gen_mol.SetProp("generation_id", generation_id)
        gen_mol.SetProp("ref_n_atoms", str(ref_n_atoms))
        gen_mol.SetProp("variance", str(variance))
        gen_mol.SetProp("valid_samples_in_generation", str(valid_samples))
        gen_mol.SetProp("requested_samples", str(requested_samples))
        gen_mol.SetProp("generation_sample_id", f"{i+1} out of {valid_samples}")
        writer.write(gen_mol)
    return None


def collect_stats(outfile_path: str):
    return None


# CORE SCRIPT
# ---------------------------------
dataset_supplier = Chem.SDMolSupplier(DATASET)
output_writer = Chem.SDWriter(OUTPUT_SDF)
ff_writer = Chem.SDWriter(FF_OUTPUT_SDF)

generation_start = time.time()
counter = 0
for r_mol in tqdm(dataset_supplier):
    try:
        counter += 1
        g_samples, fixed_fragment = ff_generate_molecules(r_mol)

        log_samples_with_metadata(generated_samples=g_samples,
                                  ref_mol=r_mol,
                                  fixed_fragment=fixed_fragment,
                                  generation_id=f"generation_{counter}",
                                  writer=output_writer,
                                  fragment_writer=ff_writer,
                                  )
    except:
        pass

generation_time = round(time.time() - generation_start, 2)

print(f"Total Generation time {generation_time} sec")








