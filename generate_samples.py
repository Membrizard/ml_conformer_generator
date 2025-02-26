import time

from ml_conformer_generator.ml_conformer_generator import (
    MLConformerGenerator,
    evaluate_samples,
)
from rdkit import Chem

device = "cuda"
generator = MLConformerGenerator(device=device)
source_path = "./data/full_15_39_atoms_conf_chembl.inchi"
n_samples = 200
max_variance = 2

references = Chem.SDMolSupplier("./data/1000_ccdc_validation_set.sdf")
sd_writer = Chem.SDWriter("100k_generated_samples.sdf")
n_ref = len(references)

for i, reference in enumerate(references):
    print(f"Analysing samples for reference compound {i + 1} of {n_ref}")
    ref_name = reference.GetProp("_Name")
    reference = Chem.RemoveHs(reference)
    ref_n_atoms = reference.GetNumAtoms()

    samples = generator.generate_conformers(
        reference_conformer=reference, n_samples=n_samples, variance=max_variance
    )

    for sample in samples:
        sample.SetProp("reference_structure", f"{ref_name}")
        sd_writer.write(sample)

