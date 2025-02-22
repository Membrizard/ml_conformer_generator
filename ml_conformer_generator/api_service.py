import os
import uuid
import logging

from fastapi import FastAPI, UploadFile, Depends, File
from pydantic import BaseModel, Field
from rdkit import Chem

from cheminformatics import evaluate_samples
from conformer_generator import MLConformerGenerator


VERSION = "0.0.1"

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


class GenerationRequest(BaseModel):
    n_samples: int
    variance: int


@app.post("/generate_molecules")
async def generate_molecules(
    file: UploadFile = File(...),
    generation_request: GenerationRequest = Depends(),
) -> dict:
    """
    Generate molecules based on the 3D shape of a reference molecule.

    - file: A `.mol`, `.xyz`, or `.pdb` file containing the reference molecule.
            The molecule should have 15–39 heavy atoms. binary
    - generation_request:
        - n_samples: The number of new molecules to generate. int
        - variance: A threshold controlling the range of number of heavy atoms in the generated molecules -
                    n_atoms = number of atoms in the reference +- variance. int
     - returns:
        A JSON-like structure containing:
        - **aligned_reference**: The original reference molecule, aligned to its principal frame using shape quadrupole.
        - **generated_molecules**: A list of newly generated molecules.

    :rtype: dict
    """

    logger.info("Uploading reference structure")

    os.makedirs(TEMP_FOLDER, exist_ok=True)
    file_path = f"{TEMP_FOLDER}/{str(uuid.uuid4())}.reference"

    try:
        with open(file_path, "w+") as f:
            content = await file.read()
            f.write(content.decode("utf-8"))

        ref_mol = Chem.MolFromMolFile(file_path)

        samples = generator.generate_conformers(
            reference_conformer=ref_mol, n_samples=generation_request.n_samples, variance=generation_request.variance
        )
        aligned_ref, std_samples = evaluate_samples(ref_mol, samples)

        results = {"aligned_reference": aligned_ref, "generated_molecules": std_samples}
        error = None

    except Exception as e:
        results = None
        error = str(e)
        pass
    finally:
        os.remove(file_path)

    return {"result": results, "errors": error}


if __name__ == "__main__":
    import uvicorn

    logger.info("--- Starting server ---")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
