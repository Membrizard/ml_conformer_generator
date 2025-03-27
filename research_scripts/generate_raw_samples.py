import os
import pickle
import time

import torch
from rdkit import Chem

from ml_conformer_generator.ml_conformer_generator import (
    MLConformerGenerator, evaluate_samples)
from ml_conformer_generator.ml_conformer_generator.utils import \
    get_context_shape

device = "cuda"
generator = MLConformerGenerator(device=device)
source_path = "./data/full_15_39_atoms_conf_chembl.inchi"
n_samples = 100
max_variance = 2

references = Chem.SDMolSupplier("./data/1000_ccdc_validation_set.sdf")
sd_writer = Chem.SDWriter("100k_generated_samples.sdf")
n_ref = len(references)
os.makedirs("./raw_samples", exist_ok=True)

for i, reference in enumerate(references):
    print(f"Generating raw samples for reference compound {i + 1} of {n_ref}")
    ref_name = reference.GetProp("_Name")
    reference = Chem.RemoveHs(reference)
    ref_n_atoms = reference.GetNumAtoms()

    conf = reference.GetConformer()
    ref_coord = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    virtual_com = torch.mean(ref_coord, dim=0)
    ref_coord = ref_coord - virtual_com

    ref_context, aligned_coord = get_context_shape(ref_coord)

    samples = generator.edm_samples(
        reference_context=ref_context,
        n_samples=n_samples,
        min_n_nodes=ref_n_atoms - max_variance,
        max_n_nodes=ref_n_atoms + max_variance,
        fix_noise=False,
    )

    # pickle
    with open(f"./raw_samples/{ref_name}_100_samples.pkl", "wb") as f:
        pickle.dump(samples, f)
