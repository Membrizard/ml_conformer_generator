from typing import List

import torch
from rdkit import Chem

from .adj_mat_seer import AdjMatSeer
from .egnn import EGNNDynamics
from .equivariant_diffusion import EquivariantDiffusion, PredefinedNoiseSchedule
from .utils import (
    ATOM_DECODER,
    CONTEXT_NORMS,
    DIMENSION,
    MAX_N_NODES,
    MIN_N_NODES,
    NUM_BOND_TYPES,
    get_context_shape,
    prepare_adj_mat_seer_input,
    prepare_edm_input,
    redefine_bonds,
    samples_to_rdkit_mol,
    standardize_mol,
)


class MLConformerGenerator(torch.nn.Module):
    """
    ML pipeline interface to generates novel molecules based on the 3D shape of a given reference molecule
    or an arbitrary context (principal components of MOI tensor).
    """

    def __init__(
        self,
        diffusion_steps: int = 100,
        device: torch.device = torch.device("cpu"),
        dimension: int = DIMENSION,
        num_bond_types: int = NUM_BOND_TYPES,
        min_n_nodes: int = MIN_N_NODES,
        max_n_nodes: int = MAX_N_NODES,
        context_norms: dict = CONTEXT_NORMS,
        atom_decoder: dict = ATOM_DECODER,
        edm_weights: str = "./edm_moi_chembl_15_39.pt",
        adj_mat_seer_weights: str = "./adj_mat_seer_chembl_15_39.pt",
    ):
        """
        Initialise the generator.

        :param diffusion_steps: Number of denoising steps - max 1000
        :param device: device to run the model on
        :param dimension: Maximal supported number of heavy atoms
        :param num_bond_types: Number of supported bond types
        :param min_n_nodes: Minimal value for number of heavy atoms in generated samples
        :param max_n_nodes: Maximal value for number of heavy atoms in generated samples
        :param context_norms: context normalisation parameters
        :param atom_decoder: decoder dict matching int atom encodings to string representations
        :param edm_weights: path to Equivariant Diffusion model state dict
        :param adj_mat_seer_weights: path to AdjMatSeer model state dict
        """
        super().__init__()

        self.device = device

        self.dimension = dimension

        self.context_norms = {
            key: torch.tensor(value) for key, value in context_norms.items()
        }

        self.atom_decoder = atom_decoder

        self.min_n_nodes = min_n_nodes
        self.max_n_nodes = max_n_nodes

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
            )["state_dict"]
        )

        adj_mat_seer.load_state_dict(
            torch.load(
                adj_mat_seer_weights,
                map_location=device,
            )["state_dict"]
        )

        # Update denoising steps for the Equivarinat Diffusion
        generative_model.gamma = PredefinedNoiseSchedule(
            timesteps=diffusion_steps, precision=1e-5
        )

        generative_model.time_steps = torch.flip(
            torch.arange(0, diffusion_steps, device=device), dims=[0]
        )

        generative_model.T = diffusion_steps
        # ----------------------------

        generative_model.to(device)
        adj_mat_seer.to(device)

        generative_model.eval()
        adj_mat_seer.eval()

        self.generative_model = generative_model
        self.adj_mat_seer = adj_mat_seer

    @torch.no_grad()
    def edm_samples(
        self,
        reference_context: torch.Tensor,
        n_samples: int = 100,
        max_n_nodes: int = 32,
        min_n_nodes: int = 25,
    ) -> List[Chem.Mol]:
        """
        Generates initial samples using generative diffusion model
        :param reference_context: reference context - tensor of shape (3)
        :param n_samples: number of samples to be generated
        :param max_n_nodes: the maximal number of heavy atoms in the among requested molecules
        :param min_n_nodes: the minimal number of heavy atoms in the among requested molecules
        :return: a list of generated samples, without atom adjacency as RDkit Mol objects
        """

        # Make sure that number of atoms of generated samples is within requested range
        if min_n_nodes < self.min_n_nodes:
            min_n_nodes = self.min_n_nodes

        if max_n_nodes > self.max_n_nodes:
            max_n_nodes = self.max_n_nodes

        node_mask, edge_mask, batch_context = prepare_edm_input(
            n_samples=n_samples,
            reference_context=reference_context,
            context_norms=self.context_norms,
            min_n_nodes=min_n_nodes,
            max_n_nodes=max_n_nodes,
            device=self.device,
        )
        x, h = self.generative_model(
            node_mask,
            edge_mask,
            batch_context,
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
        optimise_geometry: bool = True,
    ) -> List[Chem.Mol]:
        """
        Main method to generate samples from either reference molecule or an arbitrary context.
        :param reference_conformer: A 3D conformer of a reference molecule as an RDKit Mol object
        :param n_samples: number of molecules to generate
        :param variance: int - variation in number of heavy atoms for generated molecules from reference
        :param reference_context: Arbitrary Reference context if applicable, instead of reference_conformer
        :param n_atoms: Reference number of atoms when generating using arbitrary context
        :param optimise_geometry: If true will apply constrained MMFF94 geometry optimisation to generated molecules
        :return: A list of valid standardised generated molecules as RDKit Mol objects.
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

    def forward(
        self,
        reference_conformer: Chem.Mol = None,
        n_samples: int = 10,
        variance: int = 2,
        reference_context: torch.Tensor = None,
        n_atoms: int = None,
        optimise_geometry: bool = True,
    ) -> List[Chem.Mol]:
        out = self.generate_conformers(
            reference_conformer,
            n_samples,
            variance,
            reference_context,
            n_atoms,
            optimise_geometry,
        )

        return out
