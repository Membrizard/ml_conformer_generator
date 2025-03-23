from rdkit import Chem
import torch
import random

from .compilable_egnn import EGNNDynamics
from .compilable_equivariant_diffusion import EquivariantDiffusion
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

import torch_tensorrt


class MLConformerGenerator(torch.nn.Module):
    """
    A model that generates novel molecules based on the 3D shape of a given reference molecule.
    """

    def __init__(
        self,
        device: torch.device = "cpu",
        dimension: int = DIMENSION,
        num_bond_types: int = NUM_BOND_TYPES,
        edm_weights: str = "./ml_conformer_generator/ml_conformer_generator/weights/compilable_weights/"
        "compilable_edm_moi_chembl_15_39.weights",
        adj_mat_seer_weights: str = "./ml_conformer_generator/ml_conformer_generator/weights/compilable_weights/"
        "compilable_adj_mat_seer_chembl_15_39.weights",
        compile: bool = True,
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

        self.min_n_nodes = 15
        self.max_n_nodes = 39

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

        adj_mat_seer = AdjMatSeer(
            dimension=dimension,
            n_hidden=2048,
            embedding_dim=64,
            num_embeddings=36,
            num_bond_types=num_bond_types,
            device=device,
        )

        generative_model.load_state_dict(
            torch.load(
                edm_weights,
                map_location=device,
            )
        )

        adj_mat_seer.load_state_dict(
            torch.load(
                adj_mat_seer_weights,
                map_location=device,
            )
        )

        generative_model.to(device)
        adj_mat_seer.to(device)

        generative_model.eval()
        adj_mat_seer.eval()

        if compile:
            # TorchScript
            # self.generative_model = torch.jit.script(generative_model)
            # self.adj_mat_seer = torch.jit.script(adj_mat_seer)
            self.generative_model = torch.compile(generative_model, backend="inductor")
            self.adj_mat_seer = torch.compile(adj_mat_seer, backend="inductor")

        else:
            self.generative_model = generative_model
            self.adj_mat_seer = adj_mat_seer

    @torch.no_grad()
    def edm_samples(
        self,
        reference_context,
        n_samples=100,
        max_n_nodes=32,
        min_n_nodes=25,
        # fix_noise=False,
    ):
        """
        Generates initial samples using generative diffusion model
        :param reference_context: reference context - tensor of shape (3)
        :param n_samples: number of samples to be generated
        :param max_n_nodes:
        :param min_n_nodes:
        :param fix_noise:
        :return: a list of generated samples, without atom adjacency as RDkit Mol objects
        """
        # Create a random list of sizes between min_n_nodes and max_n_nodes of length n_samples
        nodesxsample = []

        # Make sure that number of atoms of generated samples is within requested range
        if min_n_nodes < self.min_n_nodes:
            min_n_nodes = self.min_n_nodes

        if max_n_nodes > self.max_n_nodes:
            max_n_nodes = self.max_n_nodes

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

        # print(f"edge_mask - {edge_mask.size()}")
        # print("batch_size")
        # print(batch_size)
        # print("max_n_nodes")
        # print(max_n_nodes)
        # print("node_mask")
        # print(node_mask)




        x, h = self.generative_model(
            batch_size,
            max_n_nodes,
            node_mask,
            edge_mask,
            batch_context,
            # fix_noise=fix_noise,
        )

        mols = samples_to_rdkit_mol(
            positions=x, one_hot=h, node_mask=node_mask, atom_decoder=self.atom_decoder
        )

        return mols

    @torch.no_grad()
    def generate_conformers(
        self,
        reference_conformer: Chem.Mol = None,
        n_samples: int = 10,
        variance: int = 2,
        reference_context: torch.Tensor = None,
        n_atoms: int = None,
        # fix_noise: bool = False,
        optimise_geometry: bool = True,
    ) -> list[Chem.Mol]:
        """

        :param reference_conformer:
        :param n_samples:
        :param variance:
        :param reference_context:
        :param n_atoms:
        # :param fix_noise:
        :param optimise_geometry:
        :return: A list of valid standardised generated molecules as RDkit Mol objects.
        """
        if reference_conformer:
            # Ensure the initial mol is stripped off Hs
            reference_conformer = Chem.RemoveHs(reference_conformer)
            ref_n_atoms = reference_conformer.GetNumAtoms()
            conf = reference_conformer.GetConformer()
            ref_coord = torch.tensor(conf.GetPositions(), dtype=torch.float32)

            # move coord to center
            virtual_com = torch.mean(ref_coord, dim=0)
            ref_coord = ref_coord - virtual_com

            ref_context, aligned_coord = get_context_shape(ref_coord)

        elif reference_context is not None:
            if n_atoms:
                ref_n_atoms = n_atoms
            else:
                raise ValueError(
                    "Reference Number of Atoms should be provided, when generating samples using context."
                )

            ref_context = reference_context

        else:
            raise ValueError(
                "Either a reference RDkit Mol object or context as torch.Tensor should be provided for generation."
            )

        edm_samples = self.edm_samples(
            reference_context=ref_context,
            n_samples=n_samples,
            min_n_nodes=ref_n_atoms - variance,
            max_n_nodes=ref_n_atoms + variance,
            # fix_noise=fix_noise,
        )

        (
            el_batch,
            dm_batch,
            b_adj_mat_batch,
            canonicalised_samples,
        ) = prepare_adj_mat_seer_input(
            mols=edm_samples,
            n_samples=n_samples,
            dimension=self.dimension,
            device=self.device,
        )

        adj_mat_batch = self.adj_mat_seer(
            elements=el_batch, dist_mat=dm_batch, adj_mat=b_adj_mat_batch
        )

        adj_mat_batch = adj_mat_batch.to("cpu")

        # Append generated bonds and standardise existing samples
        optimised_conformers = []

        for i, adj_mat in enumerate(adj_mat_batch):
            f_mol = redefine_bonds(canonicalised_samples[i], adj_mat)
            std_mol = standardize_mol(mol=f_mol, optimize_geometry=optimise_geometry)
            if std_mol:
                optimised_conformers.append(std_mol)

        return optimised_conformers
