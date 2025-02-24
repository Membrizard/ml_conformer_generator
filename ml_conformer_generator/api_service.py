import logging
from time import time

from fastapi import FastAPI, UploadFile, Depends, File
from pydantic import BaseModel, Field
from rdkit import Chem

from ml_conformer_generator import MLConformerGenerator, evaluate_samples


VERSION = "0.0.2"

TEMP_FOLDER = "./structures"
app = FastAPI(
    title=f"ML Conformer Generator Service ver {VERSION}",
    description=f"A service that generates novel molecules based on the 3D shape of a given reference molecule. {VERSION}",
)

# logger = logging.getLogger(__name__)
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

# Initiate the Generator
device = "cpu"
generator = MLConformerGenerator(device=device)


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
    errors: str = None


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
        elif ref_block == "mol2":
            ref_mol = Chem.MolFromMol2Block(ref_block)
        elif ref_block == "pdb":
            ref_mol = Chem.MolFromPDBBlock(ref_block)
        elif ref_block == "xyz":
            ref_mol = Chem.MolFromXYZBlock(ref_block)
        else:
            raise ValueError("Unsupported molecule file type.")

        logger.info("Starting Generation")
        start = time()
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

        gen_mols = []
        for sample in std_samples:
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

    logger.info("--- Starting server ---")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
