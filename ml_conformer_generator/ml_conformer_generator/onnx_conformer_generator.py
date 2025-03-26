from typing import List

import numpy
from rdkit import Chem


from .utils import (
    DIMENSION,
    NUM_BOND_TYPES,
    MIN_N_NODES,
    MAX_N_NODES,
    CONTEXT_NORMS,
    ATOM_DECODER,
    get_context_shape,
    prepare_adj_mat_seer_input,
    prepare_edm_input,
    redefine_bonds,
    samples_to_rdkit_mol,
    standardize_mol,
)


class MLConformerGeneratorONNX:
    """
    PyTorch - free ONNX implementation.
    pipeline interface to generates novel molecules based on the 3D shape of a given reference molecule
    or an arbitrary context (principal components of MOI tensor).
    """

    def __init__(
        self,
        device: str = "cpu",
        min_n_nodes: int = MIN_N_NODES,
        max_n_nodes: int = MAX_N_NODES,
        context_norms: dict = CONTEXT_NORMS,
        atom_decoder: dict = ATOM_DECODER,
        edm_onnx: str = "./ml_conformer_generator/ml_conformer_generator/weights/edm_moi_chembl_15_39.weights",
        adj_mat_seer_onnx: str = "./ml_conformer_generator/ml_conformer_generator/weights/adj_mat_seer_chembl_15_39.weights",
    ):

        self.device = device


        self.context_norms = context_norms

        self.atom_decoder = atom_decoder

        self.min_n_nodes = min_n_nodes
        self.max_n_nodes = max_n_nodes

        self.generative_model = generative_model
        self.adj_mat_seer = adj_mat_seer

    def edm_samples(
        self,
        reference_context: numpy.ndarray,
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
            n_samples,
            max_n_nodes,
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
