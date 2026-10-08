# Model Weights

All trained weights are hosted on Hugging Face:

> [https://huggingface.co/Membrizard/ml_conformer_generator](https://huggingface.co/Membrizard/ml_conformer_generator)

---

## Available Weights

| File | Format | Model | Notes |
|---|---|---|---|
| `edm_moi_chembl_15_39.pt` | PyTorch | EDM (core, 24M params) | Default for `MLConformerGenerator` |
| `adj_mat_seer_chembl_15_39.pt` | PyTorch | AdjMatSeer (core, 22M params) | Default for `MLConformerGenerator` |
| `small_edm_moi_chembl_15_39.pt` | PyTorch | EDM (distilled, 8.9M params) | Lower memory, faster |
| `small_adj_mat_seer_obabel_15_39.pt` | PyTorch | AdjMatSeer (distilled, 1.6M params) | Fine-tuned to match OpenBabel bond perception; higher validity |
| `edm_moi_chembl_6_39_fragments.pt` | PyTorch | EDM (fragments, 6–39 atoms) | Generator model for [Inertial Fragment Matching](../python/5_inertial_fragment_matching.md) |
| `edm_moi_chembl_15_39_inpaint.pt` | PyTorch | EDM (inpainting variant) | Alternative weights for fixed-fragment generation |
| `egnn_chembl_15_39.onnx` | ONNX | EGNN denoiser (core) | Default for `MLConformerGeneratorONNX` and the JS package |
| `adj_mat_seer_chembl_15_39.onnx` | ONNX | AdjMatSeer (core) | Default for `MLConformerGeneratorONNX` and the JS package |
| `small_egnn_chembl_15_39.onnx` | ONNX | EGNN denoiser (distilled) | |
| `small_adj_mat_seer_obabel_15_39.onnx` | ONNX | AdjMatSeer (distilled) | |

The PyTorch generator infers the hidden width and base number of diffusion timesteps from the checkpoint, so full and distilled weights are interchangeable — only the file names differ.

---

## Python: Automatic Resolution

Both `MLConformerGenerator` and `MLConformerGeneratorONNX` resolve weight files through a built-in `WeightsManager`. For every weights argument the following order applies:

1. If the value is an existing **local path**, it is used as is.
2. Otherwise the file is looked up in the **local cache** (`~/.mlconfgen_weights` by default).
3. If not cached, the file is **downloaded** from the Hugging Face repository into the cache. `huggingface_hub` is installed automatically if missing.

```python
from mlconfgen import MLConformerGenerator

# Default core weights, downloaded on first run
model = MLConformerGenerator()

# Distilled weights, by name
model = MLConformerGenerator(
    edm_weights="small_edm_moi_chembl_15_39.pt",
    adj_mat_seer_weights="small_adj_mat_seer_obabel_15_39.pt",
)

# Local files
model = MLConformerGenerator(
    edm_weights="./checkpoints/my_edm.pt",
    adj_mat_seer_weights="./checkpoints/my_seer.pt",
)
```

Downloads are protected by a lock file so that several processes starting concurrently do not download the same file twice.

### Listing and clearing

```python
model.list_weights()
# {'remote': ['adj_mat_seer_chembl_15_39.pt', 'edm_moi_chembl_15_39.pt', ...],
#  'local':  ['edm_moi_chembl_15_39.pt', 'adj_mat_seer_chembl_15_39.pt']}

model.clear_cache()   # removes everything under the cache directory
```

`list_weights()` filters by format — `.pt` for the PyTorch generator, `.onnx` for the ONNX generator.

### Custom cache directory

```python
from mlconfgen.utils import WeightsManager

wm = WeightsManager(cache_dir="/shared/mlconfgen_weights")
path = wm.resolve("edm_moi_chembl_15_39.pt")           # -> Path to the file
path = wm.resolve("edm_moi_chembl_15_39.pt", force_download=True)
```

---

## JavaScript: Local Files

The npm package does not download weights. Download the ONNX files from Hugging Face and pass them to `createGenerator` as file paths (Node) or `Uint8Array` buffers (browser):

```js
const gen = await createGenerator({
  ort,
  egnnOnnx: "./egnn_chembl_15_39.onnx",
  adjMatSeerOnnx: "./adj_mat_seer_chembl_15_39.onnx",
});
```

See [Browser Usage](../javascript/3_browser.md) for loading weights from a user's machine without re-hosting them.
