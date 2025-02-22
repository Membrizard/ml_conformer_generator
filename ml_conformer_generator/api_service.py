import os
import uuid
import logging
import gzip
from typing import Any, Dict


from fastapi import FastAPI, UploadFile, Depends, File
from pydantic import BaseModel, Field



VERSION = "0.0.1"

TEMP_FOLDER = "./structures"
app = FastAPI(
    title=f"ML Conformer Generator service ver {VERSION}",
    description=f"A tool for generation of molecules similar to a reference shape ver {VERSION}",
)

# logger = logging.getLogger(__name__)
logger = logging.getLogger('uvicorn')
logger.setLevel(logging.INFO)


class GenerationRequest(BaseModel):
    n_samples: int
    variance: int = False


@app.post("/generate_molecules")
async def generate_molecules(file: UploadFile = File(...),
    generation_request: GenerationRequest = Depends()) -> dict:
    """
    Generate molecules based on the 3D shape of a reference molecule.

    :param file:
        A `.mol`, `.xyz`, or `.pdb` file containing the reference molecule.
        The molecule should have 15–39 heavy atoms.
    :type file: file

    :param n_samples:
        The number of new molecules to generate.
    :type n_samples: int

    :param variance:
        A threshold controlling the range of number of heavy atoms in the generated molecules -
        n_atoms = number of atoms in the reference +- variance

    :type variance: int

    :returns:
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


    except Exception as e:
        results = None
        error = str(e)
        pass
    finally:
        os.remove(file_path)

    return {"result": results, "errors": error}

