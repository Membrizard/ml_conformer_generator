import torch
import random

from .egnn import EGNNDynamics
from .equivariant_diffusion import EquivariantDiffusion
from .adj_mat_seer import AdjMatSeer

from .utils import (
    samples_to_rdkit_mol,
    get_context_shape,
    DIMENSION,
    NUM_BOND_TYPES,
    prepare_adj_mat_seer_input,
    redefine_bonds,
    standardize_mol,
)


class MLConformerGenerator(torch.nn.Module):
    """"""

    def __init__(
        self,
        weights_path,
        device,
        dimension: int = DIMENSION,
        num_bond_types: int = NUM_BOND_TYPES,
    ):
        super().__init__()

        self.device = device

        self.dimension = dimension

        self.context_norms = {
            "mean": torch.tensor([105.0766, 473.1938, 537.4675]),
            "mad": torch.tensor([52.0409, 219.7475, 232.9718]),
        }

        self.atom_decoder = {
            0: "C",
            1: "N",
            2: "O",
            3: "F",
            4: "P",
            5: "S",
            6: "Cl",
            7: "Br",
        }

        net_dynamics = EGNNDynamics(
            in_node_nf=9,
            context_node_nf=3,
            hidden_nf=420,
            device=device,
        )

        self.generative_model = EquivariantDiffusion(
            dynamics=net_dynamics,
            in_node_nf=8,
            timesteps=1000,
            noise_precision=1e-5,
        )

        self.adj_mat_seer = AdjMatSeer(
            dimension=dimension,
            n_hidden=2048,
            embedding_dim=64,
            num_embeddings=36,
            num_bond_types=num_bond_types,
            device=device,
        )

        # self.load_state_dict()

        self.generative_model.to(device)
        self.adj_mat_seer.to(device)

    def edm_samples(
        self,
        reference_context,
        n_samples=100,
        max_n_nodes=39,
        min_n_nodes=25,
        fix_noise=False,
    ):
        """
        Generates initial samples using generative diffusion model
        :param reference_context:
        :param n_samples:
        :param max_n_nodes:
        :param min_n_nodes:
        :param fix_noise:
        :return:
        """
        # Create a random list of sizes between min_n_nodes and max_n_nodes of length n_samples
        nodesxsample = []

        for n in range(n_samples):
            nodesxsample.append(random.randint(min_n_nodes, max_n_nodes))

        nodesxsample = torch.tensor(nodesxsample)

        batch_size = nodesxsample.size(0)

        node_mask = torch.zeros(batch_size, max_n_nodes)
        for i in range(batch_size):
            node_mask[i, 0 : nodesxsample[i]] = 1

        # Compute edge_mask

        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(
            self.device
        )
        node_mask = node_mask.unsqueeze(2).to(self.device)

        normed_context = (
            (reference_context - self.context_norms["mean"]) / self.context_norms["mad"]
        ).to(self.device)

        batch_context = normed_context.unsqueeze(0).repeat(batch_size, 1)

        batch_context = batch_context.unsqueeze(1).repeat(1, max_n_nodes, 1) * node_mask

        x, h = self.generative_model.sample(
            batch_size,
            max_n_nodes,
            node_mask,
            edge_mask,
            batch_context,
            fix_noise=fix_noise,
        )

        mols = samples_to_rdkit_mol(positions=x, one_hot=h, node_mask=node_mask, atom_decoder=self.atom_decoder)

        return mols

    @torch.no_grad()
    def generate_conformers(
        self,
        reference_conformer,
        n_samples: int,
        variance: int,
        fix_noise: bool = False,
        optimise_geometry: bool = True,
    ):
        ref_n_atoms = reference_conformer.GetNumAtoms()
        conf = reference_conformer.GetConformer()
        ref_coord = torch.tensor(conf.GetPositions(), dtype=torch.float32)

        # move coord to center
        virtual_com = torch.mean(ref_coord, dim=0)
        ref_coord = ref_coord - virtual_com

        ref_context, aligned_coord = get_context_shape(ref_coord)

        edm_samples = self.edm_samples(
            reference_context=ref_context,
            n_samples=n_samples,
            min_n_nodes=ref_n_atoms - variance,
            max_n_nodes=ref_n_atoms + variance,
            fix_noise=fix_noise,
        )

        el_batch, dm_batch, b_adj_mat_batch, canonicalised_samples = prepare_adj_mat_seer_input(
            mols=edm_samples,
            n_samples=n_samples,
            dimension=self.dimension,
            device=self.device,
        )

        adj_mat_batch = self.adj_mat_seer(elements=el_batch, dist_mat=dm_batch, adj_mat=b_adj_mat_batch)

        # Append generated bonds and standardise existing samples
        optimised_conformers = []

        for i, adj_mat in enumerate(adj_mat_batch):
            f_mol = redefine_bonds(canonicalised_samples[i], adj_mat)
            std_mol = standardize_mol(mol=f_mol, optimize_geometry=optimise_geometry)
            optimised_conformers.append(std_mol)

        return optimised_conformers
