from typing import List
from pathlib import Path

import numpy as np
from rdkit import Chem

from ..utils.common import apply_transform, set_conformer_positions
from ..utils.config import (
    ATOM_DECODER,
    CONTEXT_NORMS,
    DIMENSION,
    MAX_N_NODES,
    MIN_N_NODES,
)
from ..utils.mol_split import extract_fragment
from ..utils.standardizer import standardize_mol
from .equivariant_diffusion import EquivariantDiffusionONNX
from .utils import (
    align_mol_to_principal_frame_onnx,
    prepare_adj_mat_seer_input_onnx,
    prepare_edm_input_onnx,
    prepare_fragment_onnx,
    redefine_bonds_onnx,
    samples_to_rdkit_mol_onnx,
)


class MLConformerGeneratorONNX:
    """
    PyTorch-free ONNX-based implementation
    ML pipeline interface to generates novel molecules based on the 3D shape of a given reference molecule
    or an arbitrary context (principal components of MOI tensor).
    """

    def __init__(
        self,
        diffusion_steps: int = 100,
        dimension: int = DIMENSION,
        min_n_nodes: int = MIN_N_NODES,
        max_n_nodes: int = MAX_N_NODES,
        context_norms: dict = CONTEXT_NORMS,
        atom_decoder: dict = ATOM_DECODER,
        egnn_onnx: str | Path = "./egnn_chembl_15_39.onnx",
        adj_mat_seer_onnx: str | Path = "./adj_mat_seer_chembl_15_39.onnx",
        finetune_checkpoint_onnx: str | Path = None,
    ):
        """
        Initialise the generator.

        :param min_n_nodes: Minimal value for number of heavy atoms in generated samples
        :param max_n_nodes: Maximal value for number of heavy atoms in generated samples
        :param context_norms: context normalisation parameters
        :param atom_decoder: decoder dict matching int atom encodings to string representations
        :param egnn_onnx: path to EGNN model in the ONNX format
        :param adj_mat_seer_onnx: path to AdjMatSeer model in the ONNX format
        :param finetune_checkpoint_onnx: path to a Fine Tune Checkpoint in the ONNX format
        """
        try:
            import onnxruntime
        except ImportError as e:
            raise ImportError(
                'Failed to import onnxruntime. To resolve run `pip install "mlconfgen[onnx]"`\n'
            ) from e

        super().__init__()

        self.context_norms = {
            key: np.array(value) for key, value in context_norms.items()
        }

        self.dimension = dimension

        self.atom_decoder = atom_decoder

        self.min_n_nodes = min_n_nodes
        self.max_n_nodes = max_n_nodes

        self.generative_model = EquivariantDiffusionONNX(
            egnn_onnx=egnn_onnx,
            timesteps=diffusion_steps,
            in_node_nf=8,
            noise_precision=1e-5,
        )

        self.adj_mat_seer = onnxruntime.InferenceSession(adj_mat_seer_onnx)

        self.edm_adapter = None
        if finetune_checkpoint_onnx:
            self.edm_adapter = onnxruntime.InferenceSession(finetune_checkpoint_onnx)

    @staticmethod
    def prepare_inputs(
        reference_conformer: Chem.Mol = None,
        fixed_fragment: Chem.Mol | set = None,
        reference_context: np.ndarray = None,
        n_atoms: int = None,
    ) -> tuple[np.ndarray, int, Chem.Mol | None]:
        """
        Prepare inputs for the generation forward pass.

        Applies the necessary preprocessing and business logic to the provided generation
        inputs before they are passed to the model.

        :param reference_conformer: A 3D conformer of a reference molecule as an RDKit Mol object
        :param fixed_fragment: Fragment to fix during generation as an RDKit Mol object or
                               a set of atom idxs of reference conformer
        :param reference_context: Arbitrary Reference context if applicable, instead of reference_conformer
        :param n_atoms: Reference number of atoms when generating using arbitrary context
        :returns: A tuple containing the prepared reference context, average number of atoms,
                  and the prepared fixed fragment if provided.
        """

        if reference_conformer:
            # Ensure the initial mol is stripped off Hs
            reference_conformer = Chem.RemoveAllHs(reference_conformer)
            ref_n_atoms = reference_conformer.GetNumAtoms()
            (
                ref_context,
                shift,
                rotation,
                aligned_coord,
            ) = align_mol_to_principal_frame_onnx(reference_conformer)

            if fixed_fragment:
                if isinstance(fixed_fragment, set):
                    aligned_ref_mol = set_conformer_positions(
                        reference_conformer, aligned_coord
                    )
                    fixed_fragment = extract_fragment(aligned_ref_mol, fixed_fragment)
                elif isinstance(fixed_fragment, Chem.Mol):
                    fixed_fragment = Chem.RemoveAllHs(fixed_fragment)
                    ff_conf = fixed_fragment.GetConformer()
                    ff_coord = np.array(ff_conf.GetPositions(), dtype=np.float32)
                    ff_coord_ref_aligned = apply_transform(ff_coord, shift, rotation)
                    fixed_fragment = set_conformer_positions(
                        fixed_fragment, ff_coord_ref_aligned
                    )

        elif reference_context is not None:
            if n_atoms:
                ref_n_atoms = n_atoms
            else:
                raise ValueError(
                    "Reference Number of Atoms should be provided, when generating samples using context."
                )

            ref_context = reference_context

            if isinstance(fixed_fragment, set):
                raise ValueError(
                    "'fixed_fragment' must be a Mol object when generating from a reference context."
                )

        else:
            raise ValueError(
                "Either a reference RDkit Mol object or context as numpy.ndarray should be provided for generation."
            )

        return ref_context, ref_n_atoms, fixed_fragment

    def predict_bonds(self, edm_samples: list[Chem.Mol]) -> list[Chem.Mol]:
        """
        Predict bonds using the AdjMatSeer GCN model.
        :param edm_samples: List of RDKit molecule objects without bonds or with incorrect bonds.
        :return: List of RDKit molecule objects with predicted bonds.
        """

        (
            el_batch,
            dm_batch,
            b_adj_mat_batch,
            canonicalised_samples,
        ) = prepare_adj_mat_seer_input_onnx(
            mols=edm_samples,
            dimension=self.dimension,
        )

        adj_mat_batch = self.adj_mat_seer.run(
            None,
            {"elements": el_batch, "dist_mat": dm_batch, "adj_mat": b_adj_mat_batch},
        )[0]

        # Append generated bonds and standardise existing samples
        out_mols = []

        for i, adj_mat in enumerate(adj_mat_batch):
            f_mol = redefine_bonds_onnx(canonicalised_samples[i], adj_mat)
            out_mols.append(f_mol)

        return out_mols

    def edm_samples(
        self,
        reference_context: np.ndarray,
        n_samples: int = 100,
        max_n_nodes: int = 32,
        min_n_nodes: int = 25,
        resample_steps: int = 0,
        fixed_fragment: Chem.Mol = None,
        blend_power: int = 3,
    ) -> List[Chem.Mol]:
        """
        Generates initial samples using generative diffusion model
        :param reference_context: reference context - tensor of shape (3)
        :param n_samples: number of samples to be generated
        :param max_n_nodes: the maximal number of heavy atoms in the among requested molecules
        :param min_n_nodes: the minimal number of heavy atoms in the among requested molecules
        :param resample_steps: number of resampling steps applied for harmonisation of generation
        :param fixed_fragment: fragment to retain during generation, optional
        :param blend_power: power of polynomial blending of a fixed fragment during generation
        :return: a list of generated samples, without atom adjacency as RDkit Mol objects
        """

        # Make sure that number of atoms of generated samples is within requested range
        if min_n_nodes < self.min_n_nodes:
            min_n_nodes = self.min_n_nodes

        if max_n_nodes > self.max_n_nodes:
            max_n_nodes = self.max_n_nodes

        node_mask, edge_mask, batch_context = prepare_edm_input_onnx(
            n_samples=n_samples,
            reference_context=reference_context,
            context_norms=self.context_norms,
            min_n_nodes=min_n_nodes,
            max_n_nodes=max_n_nodes,
        )
        if fixed_fragment is None:
            x, h = self.generative_model(
                node_mask,
                edge_mask,
                batch_context,
                resample_steps,
            )

        else:
            z_known, fixed_mask = prepare_fragment_onnx(
                n_samples=n_samples,
                fragment=fixed_fragment,
                max_n_nodes=max_n_nodes,
                min_n_nodes=min_n_nodes,
            )
            x, h = self.generative_model.inpaint(
                node_mask,
                edge_mask,
                batch_context,
                z_known,
                fixed_mask,
                resample_steps,
                blend_power,
            )

        if self.edm_adapter is not None:
            x, h = self.edm_adapter.run(
                None,
                {
                    "x": x.astype(np.float32),
                    "h": h.astype(np.float32),
                    "node_mask": node_mask.astype(np.float32),
                    "edge_mask": edge_mask.astype(np.float32),
                },
            )

        mols = samples_to_rdkit_mol_onnx(
            positions=x, one_hot=h, node_mask=node_mask, atom_decoder=self.atom_decoder
        )

        return mols

    def generate_conformers(
        self,
        reference_conformer: Chem.Mol = None,
        n_samples: int = 10,
        variance: int = 2,
        reference_context: np.ndarray = None,
        n_atoms: int = None,
        optimize_geometry: bool = True,
        resample_steps: int = 0,
        fixed_fragment: Chem.Mol | set = None,
        blend_power: int = 3,
    ) -> List[Chem.Mol]:
        """
        Main method to generate samples from either reference molecule or an arbitrary context.
        :param reference_conformer: A 3D conformer of a reference molecule as an RDKit Mol object
        :param n_samples: number of molecules to generate
        :param variance: int - variation in number of heavy atoms for generated molecules from reference
        :param reference_context: Arbitrary Reference context if applicable, instead of reference_conformer
        :param n_atoms: Reference number of atoms when generating using arbitrary context
        :param optimize_geometry: If true will apply constrained MMFF94 geometry optimisation to generated molecules
        :param resample_steps: number of resampling steps applied for harmonisation of generation
                               improves generation quality, while sacrificing speed
        :param fixed_fragment: Fragment to fix during generation as an RDKit Mol object
        :param blend_power: power of the polynomial blending schedule for generation with a fixed fragment
        :return: A list of valid standardised generated molecules as RDKit Mol objects.
        """

        ref_context, ref_n_atoms, fixed_fragment = self.prepare_inputs(
            reference_conformer=reference_conformer,
            fixed_fragment=fixed_fragment,
            reference_context=reference_context,
            n_atoms=n_atoms,
        )

        edm_samples = self.edm_samples(
            reference_context=ref_context,
            n_samples=n_samples,
            min_n_nodes=ref_n_atoms - variance,
            max_n_nodes=ref_n_atoms + variance,
            resample_steps=resample_steps,
            fixed_fragment=fixed_fragment,
            blend_power=blend_power,
        )

        raw_mols = self.predict_bonds(edm_samples)

        # Append generated bonds and standardise existing samples
        optimised_conformers = []
        for f_mol in raw_mols:
            std_mol = standardize_mol(
                mol=f_mol, optimize_geometry=optimize_geometry, ifm_mode=False
            )
            if std_mol:
                optimised_conformers.append(std_mol)

        return optimised_conformers

    def __call__(
        self,
        reference_conformer: Chem.Mol = None,
        n_samples: int = 10,
        variance: int = 2,
        reference_context: np.ndarray = None,
        n_atoms: int = None,
        optimize_geometry: bool = True,
        resample_steps: int = 0,
        fixed_fragment: Chem.Mol | set = None,
        blend_power: int = 3,
    ) -> List[Chem.Mol]:
        out = self.generate_conformers(
            reference_conformer,
            n_samples,
            variance,
            reference_context,
            n_atoms,
            optimize_geometry,
            resample_steps,
            fixed_fragment,
            blend_power,
        )

        return out
