# Installation

---

## Python

Requires Python 3.10 or later. Install the package with the extra matching your preferred backend:

=== "PyTorch"

    ```bash
    pip install "mlconfgen[torch]"
    ```

    Enables `MLConformerGenerator`, Inertial Fragment Matching, RL fine-tuning and ONNX export.

=== "ONNX (torch-free)"

    ```bash
    pip install "mlconfgen[onnx]"
    ```

    Enables `MLConformerGeneratorONNX` only. For GPU inference install the GPU runtime in addition:

    ```bash
    pip install onnxruntime-gpu
    ```

=== "Both"

    ```bash
    pip install "mlconfgen[full]"
    ```

The core dependencies (`rdkit`, `numpy`) are always installed. `huggingface_hub` is installed on demand the first time weights need to be downloaded.

### Verify

```python
from mlconfgen import MLConformerGenerator

model = MLConformerGenerator(diffusion_steps=10)   # downloads default weights on first run
print(model.list_weights())
```

### Devices

The PyTorch generator runs on CPU by default. Pass a device explicitly for accelerators:

```python
import torch
from mlconfgen import MLConformerGenerator

device = (
    torch.device("cuda:0") if torch.cuda.is_available()
    else torch.device("mps:0") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

model = MLConformerGenerator(device=device)
```

---

## JavaScript

Requires Node 18 or later. The package bundles `@rdkit/rdkit`; you bring your own ONNX Runtime:

=== "Node"

    ```bash
    npm install mlconfgen onnxruntime-node
    ```

=== "Browser / WebAssembly"

    ```bash
    npm install mlconfgen onnxruntime-web
    ```

`onnxruntime-node` is an optional peer dependency — the package core is runtime-neutral and you pass whichever runtime you installed to `createGenerator` as `ort`.

### Local development

```bash
git clone https://github.com/Membrizard/ml_conformer_generator.git
cd ml_conformer_generator/js
npm install
npm test
```

---

## Next step

Both libraries need the trained weights. See [Model Weights](2_model_weights.md).
