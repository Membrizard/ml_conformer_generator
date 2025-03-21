import torch.jit

from ml_conformer_generator.ml_conformer_generator.adj_mat_seer import AdjMatSeer
from ml_conformer_generator.ml_conformer_generator.compilable_egnn import GCL, EquivariantBlock, EGNNDynamics
from ml_conformer_generator.ml_conformer_generator.compilable_equivariant_diffusion import EquivariantDiffusion

net_dynamics = EGNNDynamics(hidden_nf=420, in_node_nf=9, context_node_nf=3)

diffusion = EquivariantDiffusion(
            dynamics=net_dynamics,
            in_node_nf=8,
            timesteps=1000,
            noise_precision=1e-5,
        )

# compiled_model = torch.jit.script(diffusion)


# Dummy input data for all arguments
n_samples: int
n_nodes: int,
node_mask: torch.Tensor
edge_mask: torch.Tensor
context: torch.Tensor

# Exporting to ONNX
torch.onnx.export(
    diffusion,
    (x, y, z, node_mask, edge_mask, context),  # Tuple of inputs
    "complex_model.onnx",
    input_names=["x", "y", "z", "node_mask", "edge_mask", "context"],
    output_names=["output"],
    dynamic_axes={"x": {0: "batch_size"}, "y": {0: "batch_size"}, "z": {0: "batch_size"},
                  "node_mask": {0: "batch_size"}, "edge_mask": {0: "batch_size"}, "context": {0: "batch_size"}},
    opset_version=11,
)

