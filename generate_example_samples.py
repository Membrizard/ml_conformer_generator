from ml_conformer_generator import MLConformerGenerator
from cheminformatics import evaluate_samples
from rdkit import Chem
from rdkit.Chem import rdDistGeom
import json

from rdkit.Chem import Draw

device = "cpu"
generator = MLConformerGenerator(device=device)

example_refs = ["./generation_examples/adamantanol.mol2",
                "./generation_examples/ceyyag.mol2",
                "./generation_examples/chembl223367.mol",
                "./generation_examples/chembl3955019.mol",
                "./generation_examples/chembl4089284.mol",
                "generation_examples/crown-6_ed.mol",
                "./generation_examples/yibfeu.mol2",
                ]

for i, ref in enumerate(example_refs):

    ref_mol = Chem.MolFromMolFile(ref, removeHs=False)

    if ref_mol is None:
        ref_mol = Chem.MolFromMol2File(ref, removeHs=False)

    Chem.MolToMolFile(ref_mol, f"{ref_mol.GetProp('_Name')}.mol")

    # # Generate Samples
    # samples = generator.generate_conformers(reference_conformer=ref_mol, n_samples=100, variance=2)
    #
    # # Characterise samples
    # aligned_ref, std_samples = evaluate_samples(ref_mol, samples)
    #
    # results = {"aligned_reference": aligned_ref, "generated_molecules": std_samples}
    #
    # with open(f"./generation_examples/generation_example_{i+1}.json", "w+") as outfile:
    #     json.dump(results, outfile)
