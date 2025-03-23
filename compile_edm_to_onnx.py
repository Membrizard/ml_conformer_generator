import torch.jit
import pickle
import random
from ml_conformer_generator.ml_conformer_generator.compilable_egnn import EGNNDynamics
from ml_conformer_generator.ml_conformer_generator.compilable_equivariant_diffusion import EquivariantDiffusion
from ml_conformer_generator.ml_conformer_generator.utils import get_context_shape
from rdkit import Chem
from rdkit.Chem import rdDistGeom

import torch_tensorrt

device = "cpu"
net_dynamics = EGNNDynamics(
            in_node_nf=9,
            context_node_nf=3,
            hidden_nf=420,
            device=device,
        )

generative_model = EquivariantDiffusion(
            dynamics=net_dynamics,
            in_node_nf=8,
            timesteps=1000,
            noise_precision=1e-5,
        )

generative_model.to(device)

edm_weights = "./ml_conformer_generator/ml_conformer_generator/weights/compilable_weights/compilable_edm_moi_chembl_15_39.weights"
generative_model.load_state_dict(
            torch.load(
                edm_weights,
                map_location=device,
            )
        )


generative_model.eval()

# model = generative_model.dynamics

compiled_model = torch.compile(generative_model, backend="inductor")

#
# def prepare_dummy_input():
#     """
#     """
#     with open('x.pickle', 'rb') as handle:
#         x = pickle.load(handle)
#
#     with open('t.pickle', 'rb') as handle:
#         t = pickle.load(handle)
#
#     with open('node_mask.pickle', 'rb') as handle:
#         node_mask = pickle.load(handle)
#
#     with open('edge_mask.pickle', 'rb') as handle:
#         edge_mask = pickle.load(handle)
#
#     with open('context.pickle', 'rb') as handle:
#         context = pickle.load(handle)
#
#     return t, x, node_mask, edge_mask, context
#
#
# dummy_input = prepare_dummy_input()

# Dummy input data for all arguments - Equivariant Diffusion
# n_samples = 1
# n_nodes = 2
# node_mask = torch.ones((1, 2, 1), dtype=torch.float32, device=device)
# edge_mask = torch.zeros((4, 1), dtype=torch.float32, device=device)
# context = torch.zeros((1, 2, 3), dtype=torch.float32, device=device)
#
# # dummy_input = (elements, dist_mat, adj_mat)
# dummy_input = (n_samples, n_nodes, node_mask, edge_mask, context)

# Exporting to ONNX
# torch.onnx.export(
#         compiled_model,
#         dummy_input,  # Tuple of inputs
#         "egnn_chembl_15_39.onnx",
#         do_constant_folding=True,
#         opset_version=18,
#         export_params=True,
#         input_names=["t", "x", "node_mask", "edge_mask", "context"],
#         output_names=["output"],
#         dynamic_axes={
#                       "t": {0: "n_samples", 1: "num_nodes"},
#                       "x": {0: "n_samples", 1: "num_nodes"},
#                       "node_mask": {0: "n_samples", 1: "num_nodes"},
#                       "edge_mask": {0: "num_edges"},
#                       "context": {0: "n_samples", 1: "num_nodes"},
#                       "output": {0: "n_samples", 1: "num_nodes"},
#         },
#         verbose=True,
#     )

