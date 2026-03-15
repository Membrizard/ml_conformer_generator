import logging
import time
from rdkit import Chem, RDLogger

from tqdm import tqdm


from src.mlconfgen import evaluate_samples, MLConformerGenerator
from src.mlconfgen.inertial_fragment_matching import inertial_fragment_matching


# LOGGING
RDLogger.DisableLog('rdApp.*')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [IFM Evaluation log] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

# CONFIGURATION
# ---------------------------------
IFM = True
DATASET = "./data/smoke_test_ifm_generation_5_molecules.sdf"
N_SAMPLES = 10
VARIANCE = 1
OPTIMIZE_GEOMETRY = True
DEVICE = "mps"

GENERATOR = MLConformerGenerator(
    edm_weights="./edm_moi_chembl_6_39_fragments.pt",
    adj_mat_seer_weights="./adj_mat_seer_chembl_15_39.pt",
    device=DEVICE,
    diffusion_steps=50,  # Light generation for debugging
)
VERBOSE = False
MIN_FRAG_SIZE = 6
MAX_FRAG_SIZE = 15
DSTEPS_MERGING = 10
RESAMPLE_STEPS = 0


OUTPUT_SDF = "./data/smoke_test_ifm_out.sdf"
# ---------------------------------


def generate_molecules(ref_mol: Chem.Mol,
                       ifm: bool = IFM,
                       n_samples: int = N_SAMPLES,
                       variance: int = VARIANCE,
                       optimize_geometry: bool = OPTIMIZE_GEOMETRY,
                       generator: MLConformerGenerator = GENERATOR,
                       min_frag_size: int = MIN_FRAG_SIZE,
                       max_frag_size: int = MAX_FRAG_SIZE,
                       diffusion_steps_merging: int = DSTEPS_MERGING,
                       resample_steps: int = RESAMPLE_STEPS,
                       ):

    if ifm:
        if VERBOSE:
            logging.info("Inertial Fragment Matching happening...")

        final_mols = inertial_fragment_matching(
            reference_conformer=ref_mol,
            n_samples=n_samples,
            generator=generator,  # MLConformerGenerator object
            variance=variance,
            resample_steps=resample_steps,  # resample steps
            diffusion_steps_merging=diffusion_steps_merging,  # diffusion steps for merging approx 10% from model diffusion steps
            min_frag_size=min_frag_size,  # Minimal fragment size in number of heavy atoms
            max_frag_size=max_frag_size,  # Maximal fragment size in number of heavy atoms
            max_iter=200,  # Max iterations for molecule splitting
            verbose=VERBOSE,  # Verbose flag
            predict_bonds=True,
        )
        if VERBOSE:
            logging.info("Inertial Fragment Matching happened!")

    else:
        if VERBOSE:
            logging.info("Conventional Generation happening...")
        final_mols = generator.generate_conformers(
            reference_conformer=ref_mol,
            n_samples=n_samples,
            variance=variance,
            resample_steps=resample_steps,
            optimise_geometry=False,
        )
        if VERBOSE:
            logging.info("Conventional Generation complete!")

    _, std_samples = evaluate_samples(ref_mol, final_mols)

    return std_samples


def log_samples_with_metadata(
                generated_samples: list,
                ref_mol: Chem.Mol,
                generation_id: str,
                writer: Chem.SDWriter,
                variance: int = VARIANCE,
                requested_samples: int = N_SAMPLES,
                ) -> None:

    valid_samples = len(generated_samples)
    ref_n_atoms = ref_mol.GetNumHeavyAtoms()
    ref_name = ref_mol.GetProp("_Name")

    for i, sample in enumerate(generated_samples):
        shape_similarity_score = sample["shape_tanimoto"]
        chemical_similarity_score = sample["chemical_tanimoto"]
        gen_mol = Chem.MolFromMolBlock(sample["mol_block"])

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


# CORE SCRIPT
# ---------------------------------
dataset_supplier = Chem.SDMolSupplier(DATASET)
output_writer = Chem.SDWriter(OUTPUT_SDF)

generation_start = time.time()
for r_mol in tqdm(dataset_supplier):
    g_samples = generate_molecules(r_mol)

    log_samples_with_metadata(generated_samples=g_samples,
                              ref_mol=r_mol,
                              generation_id="Mock",
                              writer=output_writer,
                              )

generation_time = round(time.time() - generation_start, 2)

print(f"Total Generation time {generation_time} sec")








