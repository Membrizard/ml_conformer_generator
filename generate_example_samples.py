from ml_conformer_generator import MLConformerGenerator, evaluate_samples
from rdkit import Chem
import json

device = "cuda"
generator = MLConformerGenerator(device=device)

example_refs = [
    "./frontend/generation_examples/CEYYAG.mol",
    "./frontend/generation_examples/CHEMBL223367_P10000055.mol",
    "./frontend/generation_examples/CHEMBL3955019_P10000113.mol",
    "./frontend/generation_examples/CHEMBL4089284_P10000009.mol",
    "./frontend/generation_examples/CROWN6.mol",
    "./frontend/generation_examples/YIBFEU.mol",
]

for i, ref in enumerate(example_refs):
    ref_mol = Chem.MolFromMolFile(ref, removeHs=False)
    print(Chem.MolToSmiles(ref_mol))

    # Generate Samples
    samples = generator.generate_conformers(
        reference_conformer=ref_mol, n_samples=100, variance=2
    )

    # Characterise samples
    aligned_ref, std_samples = evaluate_samples(ref_mol, samples)

    results = {"aligned_reference": aligned_ref, "generated_molecules": std_samples}

    with open(
        f"./frontend/generation_examples/generation_example_{i + 1}.json", "w+"
    ) as outfile:
        json.dump(results, outfile)
