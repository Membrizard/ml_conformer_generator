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

compiled_model = torch.jit.script(diffusion)
compiled_model.save("diffusion.pt")
