import torch
from rdkit import Chem
from src.mlconfgen import MLConformerGenerator
# Load a Reference conformer

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps:0")
else:
    device = torch.device("cpu")

ref_mol = Chem.MolFromMolFile('./assets/demo_files/yibfeu.mol')
N_SAMPLES = 100
VARIANCE = 1
DIFFUSION_STEPS = 20

ref_generator = MLConformerGenerator(
                                 edm_weights="./edm_moi_chembl_15_39.pt",
                                 adj_mat_seer_weights="./adj_mat_seer_chembl_15_39.pt",
                                 device=device,
                                 diffusion_steps=DIFFUSION_STEPS,
                                )

ft_generator = MLConformerGenerator(
                                 edm_weights="./edm_moi_chembl_15_39.pt",
                                 adj_mat_seer_weights="./adj_mat_seer_chembl_15_39.pt",
                                 device=device,
                                 diffusion_steps=DIFFUSION_STEPS,
                                )

ft_generator.load_fine_tune_checkpoint('')

ref_context, ref_n_atoms, fixed_fragment = ref_generator.prepare_inputs(
    reference_conformer=ref_mol,
)

edm_samples = ref_generator.edm_samples(
    reference_context=ref_context,
    n_samples=N_SAMPLES,
    min_n_nodes=ref_n_atoms - VARIANCE,
    max_n_nodes=ref_n_atoms + VARIANCE,
    resample_steps=0,
)

prior_raw_mols = ref_generator.predict_bonds(edm_samples)
ft_raw_mols = ft_generator.predict_bonds(edm_samples)



