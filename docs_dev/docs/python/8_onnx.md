# ONNX Inference & Export

The package includes a **PyTorch-free** implementation of the generation pipeline on top of ONNX Runtime, and an exporter that converts a PyTorch generator (including a fine-tuning checkpoint) to ONNX.

---

## 1. Installation

```bash
pip install "mlconfgen[onnx]"
pip install onnxruntime-gpu      # optional, for CUDA inference
```

---

## 2. `MLConformerGeneratorONNX`

```python
from rdkit import Chem
from mlconfgen import MLConformerGeneratorONNX

model = MLConformerGeneratorONNX(
    egnn_onnx="egnn_chembl_15_39.onnx",
    adj_mat_seer_onnx="adj_mat_seer_chembl_15_39.onnx",
    diffusion_steps=100,
)

reference = Chem.MolFromMolFile("./assets/demo_files/yibfeu.mol")
samples = model.generate_conformers(reference_conformer=reference, n_samples=20, variance=2)
```

The interface mirrors the PyTorch generator:

- `generate_conformers`, `edm_samples`, `predict_bonds`, `prepare_inputs`, `list_weights`, `clear_cache`
- `reference_context` is a `numpy.ndarray` of shape `(3,)`
- Fixed fragments (`fixed_fragment`, `blend_power`, `resample_steps`) are supported
- Weights are resolved through the same [WeightsManager](../getting_started/2_model_weights.md) (local path → cache → Hugging Face)

Not available on the ONNX backend: `fine_tune`, `inertial_fragment_matching`, `ff_inertial_fragment_matching`.

### Distilled weights

```python
model = MLConformerGeneratorONNX(
    egnn_onnx="small_egnn_chembl_15_39.onnx",
    adj_mat_seer_onnx="small_adj_mat_seer_obabel_15_39.onnx",
)
```

### Fine-tuning checkpoint

```python
model = MLConformerGeneratorONNX(
    egnn_onnx="egnn_chembl_15_39.onnx",
    adj_mat_seer_onnx="adj_mat_seer_chembl_15_39.onnx",
    finetune_checkpoint_onnx="./finetune_checkpoint.onnx",
)
```

> **Note:** The ONNX adapter applies the EDM part of the checkpoint. The fine-tuned AdjMatSeer head is baked into the exported AdjMatSeer ONNX file when the exporter is run on a generator with the checkpoint loaded — export both files from the same model.

### Reproducibility

The ONNX pipeline draws noise with `numpy.random`; call `np.random.seed(n)` before generation to reproduce a run. The same seed produces identical trajectories in the [JavaScript package](../javascript/1_quick_start.md#reproducibility).

---

## 3. Export to ONNX

Export requires the PyTorch backend and `onnxscript`:

```bash
pip install "mlconfgen[torch]" onnxscript
```

```python
from mlconfgen import MLConformerGenerator
from onnx_export import export_to_onnx

model = MLConformerGenerator(device="cpu")
export_to_onnx(model)
```

This writes to the current directory:

```text
./egnn_chembl_15_39.onnx
./adj_mat_seer_chembl_15_39.onnx
```

```python
export_to_onnx(
    model,
    egnn_save_path="./egnn.onnx",
    adj_mat_seer_save_path="./adj_mat_seer.onnx",
    edm_adapter_save_path="./finetune_checkpoint.onnx",
    report=False,          # True writes Markdown export reports to ./onnx_export_reports
)
```

- Build the model on the device you export from (`device="cpu"` is recommended; otherwise name the exact device, e.g. `"cuda:0"`).
- `onnx_export` is a top-level package in the repository (not part of the `mlconfgen` wheel); run the export from a checkout.
- Distilled PyTorch weights export the same way.

### Exporting a fine-tuning checkpoint

If the generator has a checkpoint loaded, the exporter additionally writes the EDM adapter:

```python
model = MLConformerGenerator(
    finetune_checkpoint="./rl_checkpoints/best_checkpoint.pt",
    device="cpu",
)
export_to_onnx(model, edm_adapter_save_path="./finetune_checkpoint.onnx")
```

The resulting `finetune_checkpoint.onnx` is consumed by `MLConformerGeneratorONNX(finetune_checkpoint_onnx=...)` and by the JavaScript package's `finetuneCheckpointOnnx` option.
