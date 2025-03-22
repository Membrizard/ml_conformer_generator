import torch.jit

from ml_conformer_generator.ml_conformer_generator.adj_mat_seer import AdjMatSeer
from ml_conformer_generator.ml_conformer_generator.compilable_egnn import GCL, EquivariantBlock, EGNNDynamics
from ml_conformer_generator.ml_conformer_generator.compilable_equivariant_diffusion import EquivariantDiffusion

net_dynamics = EGNNDynamics(hidden_nf=420, in_node_nf=9, context_node_nf=3)
adj_mat_seer = AdjMatSeer(
            dimension=42,
            n_hidden=2048,
            embedding_dim=64,
            num_embeddings=36,
            num_bond_types=5,
            device="cpu",
        )
diffusion = EquivariantDiffusion(
            dynamics=net_dynamics,
            in_node_nf=8,
            timesteps=100,
            noise_precision=1e-5,
        )

compiled_model = torch.jit.script(adj_mat_seer)


# Dummy input data for all arguments - Equivariant Diffusion
n_samples = 1
n_nodes = 20
node_mask = torch.ones((1, 20, 1), dtype=torch.float32)
edge_mask = torch.zeros((400, 1), dtype=torch.float32)
context = torch.zeros((1, 20, 3), dtype=torch.float32)

# Dummy input data for all arguments - AdjMatSeer
elements = torch.ones((4, 42), dtype=torch.long)
dist_mat = torch.ones((4, 42, 42), dtype=torch.float32)
adj_mat = torch.ones((4, 42, 42), dtype=torch.float32)

dummy_input = (elements, dist_mat, adj_mat)
# dummy_input = (n_samples, n_nodes, node_mask, edge_mask, context)

# Exporting to ONNX
torch.onnx.export(
    compiled_model.half(),
    dummy_input.half(),  # Tuple of inputs
    "complex_model.onnx",
    input_names=["elements", "dist_mat", "adj_mat"],
    output_names=["output"],
    # dynamic_axes={"x": {0: "batch_size"}, "y": {0: "batch_size"}, "z": {0: "batch_size"},
    #               "node_mask": {0: "batch_size"}, "edge_mask": {0: "batch_size"}, "context": {0: "batch_size"}},
    # opset_version=11,
)

