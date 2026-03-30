import copy
from pathlib import Path
from typing import Callable

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.distributions import Categorical
from rdkit import Chem

from .shared_prior_agent import SharedPriorAgent
from ..adj_mat_seer import AdjMatSeer
from ..utils import bond_type_dict, prepare_adj_mat_seer_input

"""
REINVENT-Style RL finetuning of the MLConfGen
"""


class RLFineTuner:
    def __init__(
        self,
        pretrained_adj_mat_seer: AdjMatSeer,
        edm_sampler_fn: Callable[[int], list[Chem.Mol]],
        score_fn: Callable[[Chem.Mol | None], float],
        lr: float = 1e-5,
        sigma: float = 10.0,
        temperature: float = 1.0,
        n_samples_per_mol: int = 4,
        reward_clip: tuple[float, float] = (-1.0, 1.0),
        device: torch.device | str = torch.device("cpu"),
    ):
        self.device = device
        self.model = SharedPriorAgent(pretrained_adj_mat_seer).to(device)

        self.edm_sampler_fn = edm_sampler_fn
        self.score_fn = score_fn

        self.sigma = sigma
        self.temperature = temperature
        self.n_samples_per_mol = n_samples_per_mol
        self.reward_clip = reward_clip

        self.optimizer = torch.optim.AdamW(self.model.agent_resize.parameters(), lr=lr)

        self.bond_type_dict = bond_type_dict
        self.prepare_input_fn = prepare_adj_mat_seer_input

    def train_step(self, edm_samples: list[Chem.Mol]) -> dict[str, float]:
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

        losses = []
        rewards = []
        valid_flags = []
        agent_lls = []
        prior_lls = []

        batch_size = len(canonicalised_samples)

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

                reward_value = self.score_fn(sampled_mol)
                reward = torch.tensor(
                    reward_value,
                    dtype=torch.float32,
                    device=agent_log_prob.device,
                )
                reward = reward.clamp(self.reward_clip[0], self.reward_clip[1])

                with torch.no_grad():
                    prior_log_prob = bond_assignment_log_prob(
                        adj_mat=prior_adj_mat,
                        sampled_bonds=sampled_bonds,
                        n_atoms=n_atoms,
                        temperature=self.temperature,
                    )

                augmented_log_prob = prior_log_prob + self.sigma * reward
                sample_loss = (augmented_log_prob - agent_log_prob).pow(2)

                losses.append(sample_loss)
                rewards.append(reward.detach())
                valid_flags.append(
                    torch.tensor(
                        1.0 if is_valid_mol(sampled_mol) else 0.0,
                        device=agent_log_prob.device,
                    )
                )
                agent_lls.append(agent_log_prob.detach())
                prior_lls.append(prior_log_prob.detach())

        loss = torch.stack(losses).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.agent_resize.parameters(), max_norm=5.0)
        self.optimizer.step()

        rewards_t = torch.stack(rewards)
        valid_t = torch.stack(valid_flags)
        agent_lls_t = torch.stack(agent_lls)
        prior_lls_t = torch.stack(prior_lls)

        return {
            "loss": float(loss.detach().cpu()),
            "reward_mean": float(rewards_t.mean().cpu()),
            "reward_std": float(rewards_t.std(unbiased=False).cpu()),
            "valid_rate": float(valid_t.mean().cpu()),
            "agent_ll_mean": float(agent_lls_t.mean().cpu()),
            "prior_ll_mean": float(prior_lls_t.mean().cpu()),
        }

    @torch.no_grad()
    def evaluate(
        self,
        n_eval_edm_samples: int = 32,
    ) -> dict[str, float]:
        edm_samples = self.edm_sampler_fn(n_eval_edm_samples)

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

        rewards = []
        valid_flags = []

        for i, base_mol in enumerate(canonicalised_samples):
            agent_adj_mat = agent_adj_mat_batch[i]

            best_reward = None
            best_valid = 0.0

            for _ in range(self.n_samples_per_mol):
                sampled_mol, _, _ = redefine_bonds_sampled(
                    mol=base_mol,
                    adj_mat=agent_adj_mat,
                    temperature=self.temperature,
                )

                reward_value = float(self.score_fn(sampled_mol))
                valid_value = 1.0 if is_valid_mol(sampled_mol) else 0.0

                if best_reward is None or reward_value > best_reward:
                    best_reward = reward_value
                    best_valid = valid_value

            rewards.append(best_reward if best_reward is not None else -1.0)
            valid_flags.append(best_valid)

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        valid_t = torch.tensor(valid_flags, dtype=torch.float32)

        return {
            "eval_reward_mean": float(rewards_t.mean().item()),
            "eval_reward_std": float(rewards_t.std(unbiased=False).item()),
            "eval_valid_rate": float(valid_t.mean().item()),
        }

    def save_agent_head(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.agent_resize.state_dict(), path)


    # def save_full_adj_mat_seer(self, path: str | Path) -> None:
    #     path = Path(path)
    #     path.parent.mkdir(parents=True, exist_ok=True)
    #
    #     base_state = self.base_adj_mat_seer.state_dict()
    #     full_state = {k: v.detach().cpu().clone() for k, v in base_state.items()}
    #
    #     full_state["resize.weight"] = self.model.agent_resize.weight.detach().cpu().clone()
    #     full_state["resize.bias"] = self.model.agent_resize.bias.detach().cpu().clone()
    #
    #     torch.save(full_state, path)

    def execute(
        self,
        edm_sampler_fn: Callable[[int], list[Chem.Mol]],
        n_epochs: int,
        edm_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_every: int = 1,
        save_dir: str = "./checkpoints_rl",
    ) -> None:

        best_eval_reward = float("-inf")

        for epoch in tqdm(range(1, n_epochs + 1)):
            edm_samples = edm_sampler_fn(edm_batch_size)
            train_stats = self.train_step(edm_samples)

            msg = (
                f"[Epoch {epoch:04d}/{n_epochs:04d}] "
                f"loss={train_stats['loss']:.4f} "
                f"reward_mean={train_stats['reward_mean']:.4f} "
                f"valid_rate={train_stats['valid_rate']:.4f} "
                f"agent_ll={train_stats['agent_ll_mean']:.4f} "
                f"prior_ll={train_stats['prior_ll_mean']:.4f}"
            )
            print(msg)

            if epoch % eval_every == 0:
                eval_stats = self.evaluate(n_eval_edm_samples=eval_batch_size)
                print(
                    f"                 "
                    f"eval_reward_mean={eval_stats['eval_reward_mean']:.4f} "
                    f"eval_valid_rate={eval_stats['eval_valid_rate']:.4f}"
                )

                if eval_stats["eval_reward_mean"] > best_eval_reward:
                    best_eval_reward = eval_stats["eval_reward_mean"]
                    self.save_agent_head(Path(save_dir) / "best_agent_resize.pt")
                    print("                 saved new best agent head")

        self.save_agent_head(Path(save_dir) / "last_agent_resize.pt")
        return None


def is_valid_mol(mol: Chem.Mol | None) -> bool:
    if mol is None:
        return False
    try:
        test_mol = Chem.Mol(mol)
        Chem.SanitizeMol(test_mol)
        return True
    except Exception:
        return False


def bond_assignment_log_prob(
    adj_mat: torch.Tensor,
    sampled_bonds: torch.Tensor,
    n_atoms: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Log-probability of a fixed sampled lower-triangular bond assignment.
    adj_mat: [N, N, T] logits
    sampled_bonds: [E]
    """
    edge_i, edge_j = torch.tril_indices(n_atoms, n_atoms, offset=-1, device=adj_mat.device)
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

    Returns:
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
            return None, torch.zeros((), device=adj_mat.device), torch.empty(
                0, dtype=torch.long, device=adj_mat.device
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
        return None, torch.zeros((), device=adj_mat.device), torch.empty(
            0, dtype=torch.long, device=adj_mat.device
        )






