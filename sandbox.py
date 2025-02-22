import torch
from ml_conformer_generator import MLConformerGenerator

generator = MLConformerGenerator()

torch.save(
                    generator.state_dict(),
    "ml_conformer_generator/weights/ml_conformer_generator_chembl_15_39.weights",
                )



