import logging
from time import time

import torch
from fastapi import Depends, FastAPI, File, UploadFile
from pydantic import BaseModel, Field
from rdkit import Chem

from mlconfgen import MLConformerGenerator, evaluate_samples

VERSION = "2.0.0"
DIFFUSION_STEPS = 100

app = FastAPI(
    title=f"ML Conformer Generator Service ver {VERSION}",
    description=f"A service that generates novel molecules based on the shape of a given reference molecule. {VERSION}",
)

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)


class InputFile(BaseModel):
    type: str  # .pdb, .mol, .mol2, .xyz
    content: str


class GenerationRequest(BaseModel):
    reference_mol: InputFile
    n_samples: int
    variance: int


class GeneratedMolecule(BaseModel):
    mol_block: str
    shape_tanimoto: float
    chemical_tanimoto: float


class GenerationResults(BaseModel):
    aligned_reference: str = ""
    generated_molecules: list[GeneratedMolecule] = []


class GenerationResponse(BaseModel):
    results: GenerationResults
    error: str = None


# Initiate the Generator
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


@app.post("/generate_molecules")
async def generate_molecules(
    generation_request: GenerationRequest,
) -> GenerationResponse:
    """
    Generate molecules based on the 3D shape of a reference molecule.

    - file: A `.mol`, (`.mol2`) `.xyz`, or `.pdb` file containing the reference molecule.
            The molecule should have 15–39 heavy atoms. binary
    - generation_request:
        - n_samples: The number of new molecules to generate. int
        - variance: A threshold controlling the range of number of heavy atoms in the generated molecules -
                    n_atoms = number of atoms in the reference +- variance. int
     - returns:
        A JSON-like structure containing:
        - **aligned_reference**: The original reference molecule, aligned to its principal frame using shape quadrupole.
        - **generated_molecules**: A list of newly generated molecules.

    :rtype: GenerationResponse
    """
    response = GenerationResponse(
        results=GenerationResults(
            aligned_reference="",
            generated_molecules=[],
        ),
    )

    try:
        ref_block_type = generation_request.reference_mol.type
        ref_block = generation_request.reference_mol.content

        if ref_block_type == "mol":
            ref_mol = Chem.MolFromMolBlock(ref_block)
        elif ref_block_type == "mol2":
            ref_mol = Chem.MolFromMol2Block(ref_block)
        elif ref_block_type == "pdb":
            ref_mol = Chem.MolFromPDBBlock(ref_block)
        elif ref_block_type == "xyz":
            ref_mol = Chem.MolFromXYZBlock(ref_block)
        else:
            raise ValueError("Unsupported molecule file type.")

        ref_mol = Chem.RemoveHs(ref_mol)
        atom_count = ref_mol.GetNumAtoms()

        if atom_count > 39 or atom_count < 15:
            raise ValueError(
                "The reference molecule should contain at least 15 but not more than 39 heavy atoms"
            )

        logger.info("Starting Generation")
        start = time()

        # Spawn model per request to allow async handling
        generator = MLConformerGenerator(device=device, diffusion_steps=DIFFUSION_STEPS)

        samples = generator.generate_conformers(
            reference_conformer=ref_mol,
            n_samples=generation_request.n_samples,
            variance=generation_request.variance,
        )
        logger.info(f"Generation Complete in {round(time() - start, 2)} sec")
        logger.info("Starting Evaluation")
        start = time()
        aligned_ref, std_samples = evaluate_samples(ref_mol, samples)
        logger.info(f"Evaluation Complete in {round(time() - start, 2)} sec")

        def s_f(x):
            return x["shape_tanimoto"]

        std_samples.sort(key=s_f, reverse=True)

        gen_mols = []
        for i, sample in enumerate(std_samples):
            gen_mols.append(
                GeneratedMolecule(
                    mol_block=sample["mol_block"],
                    shape_tanimoto=sample["shape_tanimoto"],
                    chemical_tanimoto=sample["chemical_tanimoto"],
                )
            )

        response.results.aligned_reference = aligned_ref
        response.results.generated_molecules = gen_mols
        response.errors = None

    except Exception as e:
        response.errors = str(e)
        pass

    return response


if __name__ == "__main__":
    import uvicorn
    import os  # Add this import

    logger.info("--- Starting server ---")

    # Read PORT from environment variable, default to 8000 if not set
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
