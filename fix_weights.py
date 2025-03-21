import torch
from ml_conformer_generator.ml_conformer_generator.compilable_egnn import EGNNDynamics
from ml_conformer_generator.ml_conformer_generator.compilable_equivariant_diffusion import EquivariantDiffusion
from ml_conformer_generator.ml_conformer_generator.adj_mat_seer import AdjMatSeer


net_dynamics = EGNNDynamics(
            in_node_nf=9,
            context_node_nf=3,
            hidden_nf=420,
        )

generative_model = EquivariantDiffusion(
            dynamics=net_dynamics,
            in_node_nf=8,
            timesteps=1000,
            noise_precision=1e-5,
        )

adj_mat_seer = AdjMatSeer(
            dimension=42,
            n_hidden=2048,
            embedding_dim=64,
            num_embeddings=36,
            num_bond_types=5,
            device="cpu",
        )

generative_model.load_state_dict(
            torch.load(
                "./ml_conformer_generator/ml_conformer_generator/weights/compilable_weights/compilable_edm_moi_chembl_15_39.weights",
                map_location="cpu",
            )
        )

adj_mat_seer.load_state_dict(
            torch.load(
                "./ml_conformer_generator/ml_conformer_generator/weights/compilable_weights/compilable_adj_mat_seer_chembl_15_39.weights",
                map_location="cpu",
            )
        )