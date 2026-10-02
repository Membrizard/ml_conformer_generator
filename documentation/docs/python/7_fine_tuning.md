# RL Fine-Tuning

## Overview

`MLConformerGenerator` supports **reinforcement learning (RL)-based fine-tuning** that steers the generated distribution towards molecules scoring highly on a user-defined objective.

Two lightweight components are trained; the base EDM and most of AdjMatSeer stay frozen:

- An **EDM adapter** applied to the EDM output `(x, h)`
- The **AdjMatSeer head** (`resize` layer), so bond prediction is biased towards higher-scoring molecules as well

The result is a small, portable checkpoint that can be loaded into any generator built on the same base weights.

For a complete walk-through see `rl_fine_tuning_demo.ipynb` in the repository.

---

## Notes

* If `scoring_function` is `None`, a default objective that rewards **valid molecules** is used.
* Fine-tuning is typically run at **10 diffusion steps** for speed. A checkpoint trained at 10 steps can be used at 100 steps, but `lambda_edm_adapter` should be tuned to the step count used for training.
* Scores must be in the range `[0, 1]`; `reward_clip` enforces this.

---

## 1. Scoring Function Interface

```python
from rdkit import Chem

def scoring_function(mols: list[Chem.Mol | None]) -> list[float]:
    ...
```

The list may contain `None` for samples that failed standardisation — return `0.0` for those. Any Python callable with this signature works: QSAR models, docking wrappers, property filters.

---

## 2. Example: Fine-Tuning

```python
import torch
from rdkit import Chem, RDLogger
from mlconfgen import MLConformerGenerator

RDLogger.DisableLog("rdApp.*")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MLConformerGenerator(
    edm_weights="edm_moi_chembl_15_39.pt",
    adj_mat_seer_weights="adj_mat_seer_chembl_15_39.pt",
    diffusion_steps=10,
    device=device,
)

reference = Chem.MolFromMolFile("./assets/demo_files/ceyyag.mol")

model.fine_tune(
    scoring_function=None,          # validity objective
    reference_conformer=reference,
    variance=1,
    n_epochs=20,
    train_batch_size=16,
    eval_batch_size=16,
    learning_rate=8e-5,
    sigma=60.0,
    lambda_edm_adapter=1.5,
    lambda_edm_reg=0.01,
    temperature=1.5,
    n_samples_per_mol=8,
    eval_every=2,
    save_dir="./rl_checkpoints",
)
```

### Output

```text
./rl_checkpoints/best_checkpoint.pt
./rl_checkpoints/latest_checkpoint.pt
```

---

## 3. Parameters

### Task definition

The same inputs as `generate_conformers` define what is being optimised:

| Parameter | Default | Description |
|---|---|---|
| `reference_conformer` / `reference_context` + `n_atoms` | — | Generation target. |
| `variance` | `2` | ± heavy atoms. |
| `resample_steps` | `0` | Resampling steps during sampling. |
| `fixed_fragment`, `blend_power` | `None`, `3` | Fixed-fragment setup, if any. |

### Training

| Parameter | Default | Description |
|---|---|---|
| `scoring_function` | `None` | Objective; `None` → validity. |
| `n_epochs` | `20` | Fine-tuning epochs. |
| `train_batch_size` | `64` | EDM samples drawn per epoch. |
| `eval_batch_size` | `64` | Samples used for evaluation. |
| `learning_rate` | `1e-5` | Optimiser learning rate. |
| `sigma` | `10.0` | Reward weight in the RL loss. Higher values push harder towards the objective. |
| `lambda_edm_adapter` | `0.5` | Weight of the EDM adapter term. ~`1.5` for 10 diffusion steps, `0.5–1` for 100. |
| `lambda_edm_reg` | `0.01` | Regularisation of the adapter towards identity (`0.01–0.1`). |
| `temperature` | `1.0` | Sampling temperature for AdjMatSeer bond sampling (`1.0–1.5`). |
| `n_samples_per_mol` | `16` | Bond-matrix samples drawn per EDM molecule from AdjMatSeer logits. |
| `reward_clip` | `(0, 1.0)` | Clipping bounds for the reward. |
| `eval_every` | `5` | Evaluate (and possibly save `best_checkpoint`) every *n* epochs. |
| `save_dir` | `./fine_tuning_checkpoints` | Checkpoint directory. |
| `best_checkpoint_name` | `best_checkpoint.pt` | File name of the best checkpoint. |
| `load_best_checkpoint` | `False` | Load the best checkpoint into the model when training finishes. |
| `verbose` | `True` | Print training logs. |

---

## 4. Using a Checkpoint

```python
from mlconfgen import MLConformerGenerator

# At construction
model = MLConformerGenerator(
    edm_weights="edm_moi_chembl_15_39.pt",
    adj_mat_seer_weights="adj_mat_seer_chembl_15_39.pt",
    finetune_checkpoint="./rl_checkpoints/best_checkpoint.pt",
    diffusion_steps=100,
)

# Or later
model.load_finetune_checkpoint("./rl_checkpoints/best_checkpoint.pt")
```

Loading a checkpoint attaches the EDM adapter (applied after every `edm_samples` call) and replaces the AdjMatSeer head. Checkpoints are tied to the base weights they were trained with — a checkpoint trained on the core model will not fit the distilled AdjMatSeer head and vice versa.

The ONNX backend can use a checkpoint after [export](8_onnx.md#exporting-a-fine-tuning-checkpoint).

---

## 5. REINVENT4 Scoring

The pipeline is compatible with [REINVENT4](https://github.com/MolecularAI/REINVENT4) scoring configurations. With REINVENT4 installed:

```bash
git clone https://github.com/MolecularAI/REINVENT4.git --depth 1
cd REINVENT4 && python install.py <YOUR_PROCESSOR_TYPE> && cd ..
```

```python
from mlconfgen.rl_fine_tuning.reinvent_score_wrapper import ReinventScoreWrapper

scoring_function = ReinventScoreWrapper("./assets/demo_files/scoring_config.toml")

model.fine_tune(
    scoring_function=scoring_function,
    reference_conformer=reference,
    variance=1,
    n_epochs=100,
    train_batch_size=128,
    eval_batch_size=128,
    learning_rate=8e-5,
    sigma=128.0,
    lambda_edm_adapter=1.5,
    lambda_edm_reg=0.2,
    temperature=1.5,
    n_samples_per_mol=32,
    eval_every=5,
    save_dir="./rl_checkpoints_reinvent",
)
```

`ReinventScoreWrapper(config_path, fmt=Format.TOML)` reads the `scoring` section of a REINVENT configuration (`toml`, `json` or `yaml`) and exposes it as a scoring function.

---

## 6. Practical Tips

- Start with the validity objective and a few epochs to verify the setup, then switch to your scoring function.
- Larger `train_batch_size` and `n_samples_per_mol` stabilise the policy gradient at the cost of memory.
- Monitor the evaluation score; if it collapses, lower `sigma` or raise `lambda_edm_reg`.
- The fine-tuned bond predictor is part of the checkpoint — generation with a checkpoint always uses it.
