from typing import List, Callable

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
    align_mol_to_principal_frame,
    apply_transform,
    extract_fragment,
    prepare_adj_mat_seer_input,
    prepare_edm_input,
    prepare_fragment,
    redefine_bonds,
    samples_to_rdkit_mol,
    set_conformer_positions,
    standardize_mol,
)

from .rl_fine_tune.rl_fine_tune import RLFineTuner


class MLConformerGenerator(torch.nn.Module):
    """
    ML pipeline interface to generates novel molecules based on the 3D shape of a given reference molecule
    or an arbitrary context (principal components of MOI tensor).
    """

    def __init__(
        self,
        diffusion_steps: int = 100,
        device: torch.device | str = torch.device("cpu"),
        dimension: int = DIMENSION,
        num_bond_types: int = NUM_BOND_TYPES,
        min_n_nodes: int = MIN_N_NODES,
        max_n_nodes: int = MAX_N_NODES,
        context_norms: dict = CONTEXT_NORMS,
        atom_decoder: dict = ATOM_DECODER,
        edm_weights: str = "./edm_moi_chembl_15_39.pt",
        adj_mat_seer_weights: str = "./adj_mat_seer_chembl_15_39.pt",
        fine_tune_checkpoint: str = None,
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

        gm_state_dict = torch.load(edm_weights, map_location=device)

        if "context_norms" in gm_state_dict:
            self.context_norms = {
                key: torch.tensor(value)
                for key, value in gm_state_dict["context_norms"].items()
            }
        else:
            self.context_norms = {
                key: torch.tensor(value) for key, value in context_norms.items()
            }

        generative_model.load_state_dict(gm_state_dict["state_dict"])

        ams_state_dict = torch.load(adj_mat_seer_weights, map_location=device)
        adj_mat_seer.load_state_dict(ams_state_dict["state_dict"])
        # if fine_tune_checkpoint:
        #     self.load_fine_tune_checkpoint(fine_tune_checkpoint)

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

    @staticmethod
    def prepare_inputs(
        reference_conformer: Chem.Mol = None,
        fixed_fragment: Chem.Mol | set = None,
        reference_context: torch.Tensor = None,
        n_atoms: int = None,
    ) -> tuple[torch.Tensor, int, Chem.Mol | None]:
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
            ref_context, shift, rotation, aligned_coord = align_mol_to_principal_frame(
                reference_conformer
            )

            if fixed_fragment:
                if isinstance(fixed_fragment, set):
                    aligned_ref_mol = set_conformer_positions(
                        reference_conformer, aligned_coord
                    )
                    fixed_fragment = extract_fragment(aligned_ref_mol, fixed_fragment)
                elif isinstance(fixed_fragment, Chem.Mol):
                    fixed_fragment = Chem.RemoveAllHs(fixed_fragment)
                    ff_conf = fixed_fragment.GetConformer()
                    ff_coord = torch.tensor(ff_conf.GetPositions(), dtype=torch.float32)
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
                "Either a reference RDkit Mol object or context as torch.Tensor should be provided for generation."
            )

        return ref_context, ref_n_atoms, fixed_fragment

    @torch.inference_mode()
    def predict_bonds(self, edm_samples: list[Chem.Mol]) -> list[Chem.Mol]:
        """
        Predict bonds using the AdjMatSeer GCN model
        :param edm_samples: List of RDKit molecule objects without bonds or with incorrect bonds.
        :return: List of RDKit molecule objects with predicted bonds.
        """

        (
            el_batch,
            dm_batch,
            b_adj_mat_batch,
            canonicalised_samples,
        ) = prepare_adj_mat_seer_input(
            mols=edm_samples,
            dimension=self.dimension,
            device=self.device,
        )

        adj_mat_batch = self.adj_mat_seer(
            elements=el_batch, dist_mat=dm_batch, adj_mat=b_adj_mat_batch
        )

        adj_mat_batch = adj_mat_batch.to("cpu")

        # Append generated bonds and standardise existing samples
        out_mols = []

        for i, adj_mat in enumerate(adj_mat_batch):
            mol = redefine_bonds(canonicalised_samples[i], adj_mat)
            out_mols.append(mol)

        return out_mols

    @torch.inference_mode()
    def edm_samples(
        self,
        reference_context: torch.Tensor,
        n_samples: int = 100,
        max_n_nodes: int = 32,
        min_n_nodes: int = 25,
        resample_steps: int = 0,
        fixed_fragment: Chem.Mol = None,
        blend_power: int = 3,
    ) -> List[Chem.Mol]:
        """
        Generates initial samples using the diffusion model
        :param reference_context: reference context - tensor of shape (3)
        :param n_samples: number of samples to be generated
        :param max_n_nodes: the maximal number of heavy atoms in the among requested molecules
        :param min_n_nodes: the minimal number of heavy atoms in the among requested molecules
        :param resample_steps: number of resampling steps applied for harmonisation of generation
        :param fixed_fragment: fragment to retain during generation, optional
        :param blend_power: power of polynomial blending of a fixed fragment during generation
        :return: a list of generated samples, without atom adjacency as RDkit Mol objects
        """

        # Make sure that number of atoms of generated samples is within allowed range
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

        if fixed_fragment is None:
            x, h = self.generative_model(
                node_mask,
                edge_mask,
                batch_context,
                resample_steps,
            )
        else:
            z_known, fixed_mask = prepare_fragment(
                n_samples=n_samples,
                fixed_fragment=fixed_fragment,
                max_n_nodes=max_n_nodes,
                min_n_nodes=min_n_nodes,
                device=self.device,
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

        return x, h, node_mask, edge_mask

        # mols = samples_to_rdkit_mol(
        #     positions=x, one_hot=h, node_mask=node_mask, atom_decoder=self.atom_decoder
        # )
        #
        # return mols

    @torch.inference_mode()
    def generate_conformers(
        self,
        reference_conformer: Chem.Mol = None,
        n_samples: int = 10,
        variance: int = 2,
        reference_context: torch.Tensor = None,
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
        :param fixed_fragment: Fragment to fix during generation as an RDKit Mol object or
                               a set of atom idxs of reference conformer
        :param blend_power: power of the polynomial blending schedule for generation with a fixed fragment
        :return: A list of valid standardised generated molecules as RDKit Mol objects
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

    @torch.inference_mode()
    def forward(
        self,
        reference_conformer: Chem.Mol = None,
        n_samples: int = 10,
        variance: int = 2,
        reference_context: torch.Tensor = None,
        n_atoms: int = None,
        optimize_geometry: bool = True,
        resample_steps: int = 0,
        fixed_fragment: Chem.Mol | set = None,
        blend_power: int = 3,
    ) -> List[Chem.Mol]:
        return self.generate_conformers(
            reference_conformer=reference_conformer,
            n_samples=n_samples,
            variance=variance,
            reference_context=reference_context,
            n_atoms=n_atoms,
            optimize_geometry=optimize_geometry,
            resample_steps=resample_steps,
            fixed_fragment=fixed_fragment,
            blend_power=blend_power,
        )

    def fine_tune(
        self,
        score_function: Callable[[Chem.Mol | None], float] = None,  # This should output normalised score from (0, 1)
        reference_conformer: Chem.Mol = None,
        variance: int = 2,
        reference_context: torch.Tensor = None,
        n_atoms: int = None,
        resample_steps: int = 0,
        fixed_fragment: Chem.Mol | set = None,
        blend_power: int = 3,
        n_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-5,
        sigma: float = 10.0,
        temperature: float = 1.0,
        n_samples_per_mol: int = 4,
        reward_clip: tuple[float, float] = (0.0, 1.0),
        eval_every: int = 1,
        save_dir: str = "./rl_fine_tuning_checkpoints",
        best_checkpoint_name: str = "best_agent_resize.pt",
        load_best_checkpoint: bool = False,
    ):
        """
        Fine-Tunes the model for the specific task using a scoring function
        """

        if score_function is None:

            def score_function(x):
                return reward_clip[1]

        ref_context, ref_n_atoms, fixed_fragment = self.prepare_inputs(
            reference_conformer=reference_conformer,
            fixed_fragment=fixed_fragment,
            reference_context=reference_context,
            n_atoms=n_atoms,
        )

        min_n_nodes = ref_n_atoms - variance
        max_n_nodes = ref_n_atoms + variance

        def edm_sampler_fn(_batch_size: int) -> list[Chem.Mol]:
            """ """
            return self.edm_samples(
                reference_context=ref_context,
                n_samples=_batch_size,
                min_n_nodes=min_n_nodes,
                max_n_nodes=max_n_nodes,
                resample_steps=resample_steps,
                fixed_fragment=fixed_fragment,
                blend_power=blend_power,
            )

        def _score_fn(mol: Chem.Mol | None) -> float:
            """
            A score funtion with a validity wrapper, automatically setting score of invalid molecules to lowest possible
            """
            if mol is None:
                return -1
            try:
                test_mol = Chem.Mol(mol)
                Chem.SanitizeMol(test_mol)
                # valid_bonus = (reward_clip[0] + reward_clip[1]) / 2
                score = score_function(mol)
                return score
            except Exception:
                return -1

        # 4) Train

        trainer = RLFineTuner(
            pretrained_adj_mat_seer=self.adj_mat_seer,
            edm_sampler_fn=edm_sampler_fn,
            score_fn=_score_fn,
            lr=learning_rate,
            sigma=sigma,
            temperature=temperature,
            n_samples_per_mol=n_samples_per_mol,
            reward_clip=reward_clip,
            device=self.device,
        )

        trainer.execute(
            edm_sampler_fn=edm_sampler_fn,
            n_epochs=n_epochs,
            edm_batch_size=batch_size,
            eval_every=eval_every,
            save_dir=save_dir,
            best_checkpoint_name=best_checkpoint_name,
        )

        # if load_best_checkpoint:
        #     self.load_fine_tune_checkpoint(best_checkpoint_name)
        return None

    # def load_fine_tune_checkpoint(self, checkpoint_path: str) -> None:
    #     self.adj_mat_seer.resize.load_state_dict(torch.load(checkpoint_path))
    #     return None
