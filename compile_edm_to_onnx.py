import torch.jit

from ml_conformer_generator.ml_conformer_generator.compilable_egnn import EGNNDynamics
from ml_conformer_generator.ml_conformer_generator.compilable_equivariant_diffusion import EquivariantDiffusion

net_dynamics = EGNNDynamics(
            in_node_nf=9,
            context_node_nf=3,
            hidden_nf=420,
            device="cuda",
        )

generative_model = EquivariantDiffusion(
            dynamics=net_dynamics,
            in_node_nf=8,
            timesteps=1000,
            noise_precision=1e-5,
        )

generative_model.to("cuda")

edm_weights = "./ml_conformer_generator/ml_conformer_generator/weights/compilable_weights/compilable_edm_moi_chembl_15_39.weights"
generative_model.load_state_dict(
            torch.load(
                edm_weights,
                map_location="cuda",
            )
        )

generative_model.eval()
compiled_model = torch.jit.script(generative_model)


# Dummy input data for all arguments - Equivariant Diffusion
n_samples = 4
n_nodes = 20
node_mask = torch.ones((4, 20, 1), dtype=torch.float32, device="cuda")
edge_mask = torch.zeros((1600, 1), dtype=torch.float32, device="cuda")
context = torch.zeros((4, 20, 3), dtype=torch.float32, device="cuda")

# dummy_input = (elements, dist_mat, adj_mat)
dummy_input = (n_samples, n_nodes, node_mask, edge_mask, context)

# Exporting to ONNX
torch.onnx.export(
    compiled_model,
    dummy_input,  # Tuple of inputs
    "moi_edm_chembl_15_39.onnx",
    do_constant_folding=True,
    input_names=["n_samples", "n_nodes", "node_mask", "edge_mask", "context"],
    output_names=["x", "h"],
    dynamic_axes={
                  "node_mask": {0: "batch_size", 1: "num_nodes"},
                  "edge_mask": {0: "num_edges"},
                  "context": {0: "batch_size", 1: "num_nodes"},
                  "x": {0: "batch_size", 1: "num_nodes"},
                  "h": {0: "batch_size", 1: "num_nodes"},
    },
    opset_version=18,
)
