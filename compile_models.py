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
n_samples = 2
n_nodes = 42
node_mask = torch.zeros()
edge_mask = torch.zeros()
context = torch.zeros()

dummy_input = (n_samples, n_nodes, node_mask, edge_mask, context)

# Exporting to ONNX
torch.onnx.export(
    diffusion,
    dummy_input,  # Tuple of inputs
    "complex_model.onnx",
    input_names=["n_samples", "n_nodes", "node_mask", "edge_mask", "context"],
    output_names=["output"],
    dynamic_axes={"x": {0: "batch_size"}, "y": {0: "batch_size"}, "z": {0: "batch_size"},
                  "node_mask": {0: "batch_size"}, "edge_mask": {0: "batch_size"}, "context": {0: "batch_size"}},
    opset_version=11,
)

