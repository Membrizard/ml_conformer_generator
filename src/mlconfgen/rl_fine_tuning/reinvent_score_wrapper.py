from enum import Enum
from pathlib import Path

import numpy as np
from rdkit import Chem

from ..utils import is_valid_mol

try:
    import reinvent
    from reinvent.scoring.scorer import Scorer
    from reinvent.utils.config_parse import read_config
except ImportError as e:
    raise ImportError(
        'Failed to import reinvent. To resolve install REINVENT4: https://github.com/MolecularAI/REINVENT4/tree/main`\n'
    ) from e


class Format(str, Enum):
    TOML = "toml"
    JSON = "json"
    YAML = "yaml"


class ReinventScoreWrapper:
    """
        Wrapper around REINVENT Scorer object to simplify its usage in fine tuning
    """
    def __init__(self, config_path: str | Path, fmt: Format = Format.TOML):
        """
        :param config_path: Path to a REINVENT configuration file with scoring section specified
        :param fmt: format of the file - one of: 'toml', 'json', 'yaml' (Format.TOML, Format.JSON, Format.YAML)
        """
        config = read_config(config_path, fmt)
        self.reinvent_scorer = Scorer(config["scoring"])

    def __call__(self, mols: list[Chem.Mol | None]):
        smilies = []
        invalid_mask = []
        duplicate_mask = []

        for mol in mols:
            valid_flag = is_valid_mol(mol)
            if valid_flag == 1.0:
                invalid_mask.append(1)
                smiles = Chem.MolToSmiles(mol)
                if smiles in smilies:
                    duplicate_mask.append(0)
                else:
                    duplicate_mask.append(1)
                smilies.append(smiles)
            else:
                smilies.append("")
                invalid_mask.append(0)
                duplicate_mask.append(0)

        invalid_mask = np.array(invalid_mask, dtype=np.int8)
        duplicate_mask = np.array(invalid_mask, dtype=np.int8)

        score_results = self.reinvent_scorer(smilies, invalid_mask, duplicate_mask)
        scores = score_results.total_scores

        return scores.tolist()

