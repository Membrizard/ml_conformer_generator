import logging
from pathlib import Path
from typing import Callable

import torch
from rdkit import Chem
from torch.distributions import Categorical

from ..adj_mat_seer import AdjMatSeer
from ..utils import (ATOM_DECODER, bond_type_dict, is_valid_mol,
                     prepare_adj_mat_seer_input, redefine_bonds,
                     samples_to_rdkit_mol)
from .edm_adapter import EDMAdapter
from .shared_prior_agent import SharedPriorAgent

logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class RLFineTuner:
    """
    Reinforcement Learning Fine Tuner for MLConformerGenerator.
    """
    def __init__(
        self,
        pretrained_adj_mat_seer: AdjMatSeer,
        edm_sampler_fn: Callable[[int], list[Chem.Mol]],
        score_fn: Callable[[list[Chem.Mol | None]], list[float]],
        lr: float = 1e-5,
        lambda_adapter: float = 0.5,
        lambda_reg: float = 0.01,
        sigma: float = 10.0,
        temperature: float = 1.0,
        n_samples_per_mol: int = 4,
        reward_clip: tuple[float, float] = (0, 1.0),
        device: torch.device | str = torch.device("cpu"),
        atom_decoder: dict = ATOM_DECODER,
        bond_dict: dict = bond_type_dict,
    ):
        self.device = device

        self.model = SharedPriorAgent(pretrained_adj_mat_seer).to(device)
        self.edm_adapter = EDMAdapter(device).to(device)

        self.edm_sampler_fn = edm_sampler_fn
        self.score_fn = score_fn

        self.sigma = sigma
        self.temperature = temperature
        self.lambda_adapter = lambda_adapter
        self.lambda_reg = lambda_reg

        self.n_samples_per_mol = n_samples_per_mol
        self.reward_clip = reward_clip

        trainable_params = list(self.model.agent_resize.parameters()) + list(
            self.edm_adapter.parameters()
        )
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        self.bond_type_dict = bond_dict
        self.atom_decoder = atom_decoder
        self.prepare_input_fn = prepare_adj_mat_seer_input

    def save_checkpoint(self, path: str | Path) -> None:
        edm_adapter = self.edm_adapter.state_dict()
        gcn_head = self.model.agent_resize.state_dict()

        checkpoint = {
            "edm_adapter": edm_adapter,
            "adj_mat_seer_head": gcn_head,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        return None

    def load_checkpoint(self, path: str | Path):
        checkpoint = torch.load(path, map_location=self.device)

        self.edm_adapter.load_state_dict(checkpoint["edm_adapter"])
        self.model.agent_resize.load_state_dict(checkpoint["adj_mat_seer_head"])
        return None

    def train_step(self, x, h, node_mask, edge_mask) -> dict[str, float]:
        x = x.clone()
        h = h.clone()
        node_mask = node_mask.clone()
        edge_mask = edge_mask.clone()

        _x, _h, edm_adapter_log_probs, edm_aux = self.edm_adapter(
            x=x,
            h=h,
            node_mask=node_mask,
            edge_mask=edge_mask,
            sample=True,
        )

        x_safe = _x.clone().detach()
        h_safe = _h.clone().detach()

        edm_samples = samples_to_rdkit_mol(
            positions=x_safe,
            one_hot=h_safe,
            node_mask=node_mask,
            atom_decoder=ATOM_DECODER,
        )

        (
            el_batch,
            dm_batch,
            b_adj_mat_batch,
            canonicalised_samples,
        ) = self.prepare_input_fn(
            mols=edm_samples,
            dimension=self.model.dimension,
            device=self.device,
        )

        agent_adj_mat_batch = self.model.agent_forward(
            elements=el_batch,
            dist_mat=dm_batch,
            adj_mat=b_adj_mat_batch,
        )

        prior_adj_mat_batch = self.model.prior_forward(
            elements=el_batch,
            dist_mat=dm_batch,
            adj_mat=b_adj_mat_batch,
        )

        # losses = []
        # rewards = []
        valid_flags = []
        agent_lls = []
        prior_lls = []

        batch_size = len(canonicalised_samples)

        _mols = []

        # Move to CPU for more efficient cycle compute?
        # agent_adj_mat_batch.to('cpu')
        # prior_adj_mat_batch.to('cpu')

        for i in range(batch_size):
            base_mol = canonicalised_samples[i]
            n_atoms = base_mol.GetNumAtoms()

            agent_adj_mat = agent_adj_mat_batch[i]
            prior_adj_mat = prior_adj_mat_batch[i]

            for _ in range(self.n_samples_per_mol):
                sampled_mol, agent_log_prob, sampled_bonds = redefine_bonds_sampled(
                    mol=base_mol,
                    adj_mat=agent_adj_mat,
                    temperature=self.temperature,
                )

                with torch.no_grad():
                    prior_log_prob = bond_assignment_log_prob(
                        adj_mat=prior_adj_mat,
                        sampled_bonds=sampled_bonds,
                        n_atoms=n_atoms,
                        temperature=self.temperature,
                    )

                _mols.append(sampled_mol)
                agent_lls.append(agent_log_prob)
                prior_lls.append(prior_log_prob)

                valid_flags.append(
                    torch.tensor(is_valid_mol(sampled_mol), device=self.device)
                )

                # reward_value = self.score_fn(sampled_mol)
                # reward = torch.tensor(
                #     reward_value,
                #     dtype=torch.float32,
                #     device=agent_log_prob.device,
                # )
                # reward = reward.clamp(self.reward_clip[0], self.reward_clip[1])

                # with torch.no_grad():
                #     prior_log_prob = bond_assignment_log_prob(
                #         adj_mat=prior_adj_mat,
                #         sampled_bonds=sampled_bonds,
                #         n_atoms=n_atoms,
                #         temperature=self.temperature,
                #     )

                # augmented_log_prob = prior_log_prob + self.sigma * reward
                # sample_loss = (agent_log_prob - augmented_log_prob).pow(2)

                # losses.append(sample_loss)
                # rewards.append(reward.detach())
                # valid_flags.append(
                #     torch.tensor(
                #         1.0 if is_valid_mol(sampled_mol) else 0.0,
                #         device=agent_log_prob.device,
                #     )
                #  )
                # agent_lls.append(agent_log_prob.detach())
                # prior_lls.append(prior_log_prob.detach())

        # loss_bond = torch.stack(losses).mean().to(self.device)
        # rewards_t = torch.stack(rewards).to(self.device)
        rewards = torch.tensor(self.score_fn(_mols), dtype=torch.float32, device=self.device)
        agent_lls_t = torch.stack(agent_lls)
        prior_lls_t = torch.stack(agent_lls)

        rewards_t = rewards.clamp(self.reward_clip[0], self.reward_clip[1])
        augmented_log_prob = prior_lls_t + self.sigma * rewards
        # losses = (agent_lls_t - augmented_log_prob).pow(2)

        loss_bond =(agent_lls_t - augmented_log_prob).pow(2).mean()
        # rewards_t = torch.stack(rewards)

        rewards_per_edm = rewards_t.view(batch_size, self.n_samples_per_mol).mean(dim=1)
        advantages_edm = rewards_per_edm - rewards_per_edm.mean()

        loss_edm_adapter = -(advantages_edm.detach() * edm_adapter_log_probs).mean()
        loss_reg = edm_aux["dx_mean_l2"] + edm_aux["dh_mean_l2"]

        loss = (
            loss_bond
            + self.lambda_adapter * loss_edm_adapter
            + self.lambda_reg * loss_reg
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        valid_t = torch.stack(valid_flags)
        agent_lls_t = torch.stack(agent_lls)
        prior_lls_t = torch.stack(prior_lls)

        return {
            "loss": loss.detach().item(),
            "reward_mean": rewards_t.mean().item(),
            "reward_std": rewards_t.std(unbiased=False).item(),
            "valid_rate": valid_t.mean().item(),
            "agent_ll_mean": agent_lls_t.mean().item(),
            "prior_ll_mean": prior_lls_t.mean().item(),
        }

    @torch.no_grad()
    def evaluate(
        self,
        n_eval_edm_samples: int = 32,
    ) -> dict[str, float]:
        x, h, node_mask, edge_mask = self.edm_sampler_fn(n_eval_edm_samples)

        _x, _h, edm_adapter_log_probs, edm_aux = self.edm_adapter(
            x=x,
            h=h,
            node_mask=node_mask,
            edge_mask=edge_mask,
            sample=False,
        )

        edm_samples = samples_to_rdkit_mol(
            positions=_x, one_hot=_h, node_mask=node_mask, atom_decoder=ATOM_DECODER
        )

        baseline_samples = samples_to_rdkit_mol(
            positions=x, one_hot=h, node_mask=node_mask, atom_decoder=ATOM_DECODER
        )

        (
            el_batch,
            dm_batch,
            b_adj_mat_batch,
            canonicalised_samples,
        ) = self.prepare_input_fn(
            mols=edm_samples,
            dimension=self.model.dimension,
            device=self.device,
        )

        (
            bl_el_batch,
            bl_dm_batch,
            bl_b_adj_mat_batch,
            bl_canonicalised_samples,
        ) = self.prepare_input_fn(
            mols=baseline_samples,
            dimension=self.model.dimension,
            device=self.device,
        )

        agent_adj_mat_batch = self.model.agent_forward(
            elements=el_batch,
            dist_mat=dm_batch,
            adj_mat=b_adj_mat_batch,
        )

        baseline_adj_mat_batch = self.model.prior_forward(
            elements=bl_el_batch,
            dist_mat=bl_dm_batch,
            adj_mat=bl_b_adj_mat_batch,
        )

        # agent_scores = []
        # baseline_scores = []

        agent_valid_flags = []
        baseline_valid_flags = []

        agent_mols = []
        baseline_mols = []

        for i, base_mol in enumerate(canonicalised_samples):
            agent_adj_mat = agent_adj_mat_batch[i]
            baseline_adj_mat = baseline_adj_mat_batch[i]
            _baseline_mol = bl_canonicalised_samples[i]

            # agent_mol = redefine_bonds(mol=base_mol, adj_mat=agent_adj_mat)
            # baseline_mol = redefine_bonds(mol=_baseline_mol, adj_mat=baseline_adj_mat)

            agent_mol, agent_valid_value = _eval_op(base_mol, agent_adj_mat)
            baseline_mol, baseline_valid_value = _eval_op(_baseline_mol, baseline_adj_mat)

            agent_mols.append(agent_mol)
            baseline_mols.append(baseline_mol)

            # agent_scores.append(agent_score)
            # baseline_scores.append(baseline_score)

            agent_valid_flags.append(agent_valid_value)
            baseline_valid_flags.append(baseline_valid_value)

        agent_scores_t = torch.tensor(self.score_fn(agent_mols), dtype=torch.float32)
        baseline_scores_t = torch.tensor(self.score_fn(baseline_mols), dtype=torch.float32)

        agent_valid_t = torch.tensor(agent_valid_flags, dtype=torch.float32)
        baseline_valid_t = torch.tensor(baseline_valid_flags, dtype=torch.float32)

        f_agent_score = agent_scores_t.mean().item()
        f_baseline_score = baseline_scores_t.mean().item()

        baseline_imporv = f_agent_score - f_baseline_score

        return {
            "eval_agent_scores_mean": agent_scores_t.mean().item(),
            "eval_baseline_scores_mean": baseline_scores_t.mean().item(),
            "eval_agent_valid_rate": agent_valid_t.mean().item(),
            "eval_baseline_valid_rate": baseline_valid_t.mean().item(),
            "eval_improve_mean": baseline_imporv,
        }

    def execute(
        self,
        edm_sampler_fn: Callable[[int], list[Chem.Mol]],
        n_epochs: int,
        edm_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_every: int = 1,
        save_dir: str = "./checkpoints_rl",
        best_checkpoint_name: str = "best_agent_resize.pt",
        verbose: bool = True,
    ) -> None:
        best_eval_improv = 0
        best_checkpoint_path = f"{save_dir}/{best_checkpoint_name}"
        latest_checkpoint_path = f"{save_dir}/latest_checkpoint.pt"

        for epoch in range(1, n_epochs + 1):
            x, h, node_mask, edge_mask = edm_sampler_fn(edm_batch_size)
            train_stats = self.train_step(x, h, node_mask, edge_mask)

            msg = (
                f"[Epoch {epoch:04d}/{n_epochs:04d}] "
                f"loss={train_stats['loss']:.4f} "
                f"mean_score={train_stats['reward_mean']:.4f} "
                f"valid_rate={train_stats['valid_rate']:.4f} "
                f"agent_ll={train_stats['agent_ll_mean']:.4f} "
                f"prior_ll={train_stats['prior_ll_mean']:.4f}"
            )
            if verbose:
                logger.info(msg)

            if epoch % eval_every == 0:
                eval_stats = self.evaluate(n_eval_edm_samples=eval_batch_size)

                if verbose:
                    logger.info(
                        f"Evaluation:\n"
                        f"                            "
                        f" agent_score_mean={eval_stats['eval_agent_scores_mean']:.4f} "
                        f" baseline_score_mean={eval_stats['eval_baseline_scores_mean']:.4f}\n"
                        f"                            "
                        f" eval_agent_valid_rate={eval_stats['eval_agent_valid_rate']:.4f} "
                        f" eval_baseline_valid_rate={eval_stats['eval_baseline_valid_rate']:.4f}\n"
                        f"                            "
                        f" score_improv={eval_stats['eval_improve_mean']:.4f} "
                        f" valid_rate_improv={(eval_stats['eval_agent_valid_rate'] - eval_stats['eval_baseline_valid_rate']):.4f} "
                    )

                if eval_stats["eval_improve_mean"] > best_eval_improv:
                    best_eval_improv = eval_stats["eval_improve_mean"]
                    self.save_checkpoint(best_checkpoint_path)
                    if verbose:
                        logger.info("Saved new best checkpoint")

        self.save_checkpoint(latest_checkpoint_path)
        return None


# def is_valid_mol(mol: Chem.Mol | None) -> bool:
#     if mol is None:
#         return False
#     try:
#         test_mol = Chem.Mol(mol)
#         Chem.SanitizeMol(test_mol)
#         return True
#     except Exception:
#         return False


def bond_assignment_log_prob(
    adj_mat: torch.Tensor,
    sampled_bonds: torch.Tensor,
    n_atoms: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Log-probability of a fixed sampled lower-triangular bond assignment.
    :param adj_mat: [N, N, T] logits
    :param sampled_bonds: [E]
    :param n_atoms:
    :param temperature:
    :returns:
    """
    edge_i, edge_j = torch.tril_indices(
        n_atoms, n_atoms, offset=-1, device=adj_mat.device
    )
    edge_logits = adj_mat[edge_i, edge_j, :] / temperature
    dist = Categorical(logits=edge_logits)
    return dist.log_prob(sampled_bonds).sum()


def redefine_bonds_sampled(
    mol: Chem.Mol,
    adj_mat: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[Chem.Mol | None, torch.Tensor, torch.Tensor]:
    """
    Sample one bond assignment from adjacency logits.
    :param mol: RDkit Mol Object to write bonds to
    :param adj_mat: Adjacency matrix tensor with initial logits
    :param temperature:
    :returns:
        new_mol: RDKit mol or None
        log_prob: scalar tensor
        sampled_bonds: [E] lower-triangular sampled classes
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    n = mol.GetNumAtoms()

    try:
        i_xyz = Chem.MolToXYZBlock(mol)
        c_mol = Chem.MolFromXYZBlock(i_xyz)
        if c_mol is None:
            return (
                None,
                torch.zeros((), device=adj_mat.device),
                torch.empty(0, dtype=torch.long, device=adj_mat.device),
            )

        ed_mol = Chem.EditableMol(c_mol)

        edge_i, edge_j = torch.tril_indices(n, n, offset=-1, device=adj_mat.device)
        edge_logits = adj_mat[edge_i, edge_j, :] / temperature

        dist = Categorical(logits=edge_logits)
        sampled_bonds = dist.sample()
        log_prob = dist.log_prob(sampled_bonds).sum()

        for k in range(sampled_bonds.size(0)):
            i = int(edge_i[k].item())
            j = int(edge_j[k].item())
            bond_type = int(sampled_bonds[k].item())

            if bond_type != 0:
                ed_mol.AddBond(i, j, bond_type_dict[bond_type])

        new_mol = ed_mol.GetMol()
        return new_mol, log_prob, sampled_bonds

    except Exception:
        return (
            None,
            torch.zeros((), device=adj_mat.device),
            torch.empty(0, dtype=torch.long, device=adj_mat.device),
        )


def _eval_op(base_mol: Chem.Mol, adj_mat: torch.Tensor) -> tuple:
    mol = redefine_bonds(mol=base_mol, adj_mat=adj_mat)
    valid_value = is_valid_mol(mol)

    return mol, valid_value
