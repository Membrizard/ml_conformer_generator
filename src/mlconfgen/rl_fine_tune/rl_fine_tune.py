from pathlib import Path
from typing import Callable

import torch
from torch.distributions import Categorical
from rdkit import Chem

from .shared_prior_agent import SharedPriorAgent
from ..adj_mat_seer import AdjMatSeer
from ..utils import bond_type_dict, prepare_adj_mat_seer_input, redefine_bonds, samples_to_rdkit_mol, ATOM_DECODER
from .edm_lora import EDMLoRAPolicy


class RLFineTuner:
    """
    Reinforcement Learning Fine Tuner for MLConformer generator.


    """

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
        self.edm_adapter = EDMLoRAPolicy(device).to(device)

        self.lambda_adapter = 0.5
        self.lambda_reg = 0.01

        self.edm_sampler_fn = edm_sampler_fn
        self.score_fn = score_fn

        self.sigma = sigma
        self.temperature = temperature
        self.n_samples_per_mol = n_samples_per_mol
        self.reward_clip = reward_clip

        trainable_params = (
            list(self.model.agent_resize.parameters())
            # + list(self.model.agent_gcn4.parameters())
            + list(self.edm_adapter.parameters())
        )
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr)

        self.bond_type_dict = bond_type_dict
        self.prepare_input_fn = prepare_adj_mat_seer_input

    def save_agent_head(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.agent_resize.state_dict(), path)
        return None

    def train_step(self, x, h, node_mask, edge_mask) -> dict[str, float]:

        # Duplicate tensor from inference mode
        x = x.clone()
        h = h.clone()
        node_mask = node_mask.clone()
        edge_mask = edge_mask.clone()

        _x, _h, edm_adapter_log_probs, edm_aux = self.edm_adapter(
            x=x, h=h, node_mask=node_mask, edge_mask=edge_mask, sample=True,
        )

        x_safe = _x.clone().detach()
        h_safe = _h.clone().detach()

        edm_samples = samples_to_rdkit_mol(
            positions=x_safe, one_hot=h_safe, node_mask=node_mask, atom_decoder=ATOM_DECODER
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
                sample_loss = (agent_log_prob - augmented_log_prob).pow(2)

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

        loss_bond = torch.stack(losses).mean()

        rewards_t = torch.stack(rewards)

        rewards_per_edm = rewards_t.view(batch_size, self.n_samples_per_mol).mean(dim=1)  # [B]
        advantages_edm = rewards_per_edm - rewards_per_edm.mean()

        loss_edm_adapter = -(advantages_edm.detach() * edm_adapter_log_probs).mean()
        loss_reg = (
                edm_aux["dx_mean_l2"]
                + edm_aux["dh_mean_l2"]
        )

        loss = loss_bond + self.lambda_adapter * loss_edm_adapter + self.lambda_reg * loss_reg

        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.agent_resize.parameters(), max_norm=5.0)
        self.optimizer.step()

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
        x, h, node_mask, edge_mask = self.edm_sampler_fn(n_eval_edm_samples)

        # Duplicate tensor from inference mode
        x = x.clone()
        h = h.clone()
        node_mask = node_mask.clone()
        edge_mask = edge_mask.clone()

        _x, _h, edm_adapter_log_probs, edm_aux = self.edm_adapter(
            x=x, h=h, node_mask=node_mask, edge_mask=edge_mask, sample=False,
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

        prior_adj_mat_batch = self.model.prior_forward(
            elements=el_batch,
            dist_mat=dm_batch,
            adj_mat=b_adj_mat_batch,
        )

        baseline_adj_mat_batch = self.model.prior_forward(
            elements=bl_el_batch,
            dist_mat=bl_dm_batch,
            adj_mat=bl_b_adj_mat_batch,
        )

        agent_scores = []
        prior_scores = []
        baseline_scores = []

        agent_valid_flags = []
        prior_valid_flags = []
        baseline_valid_flags = []

        for i, base_mol in enumerate(canonicalised_samples):
            agent_adj_mat = agent_adj_mat_batch[i]
            prior_adj_mat = prior_adj_mat_batch[i]
            baseline_adj_mat = baseline_adj_mat_batch[i]
            _baseline_mol = bl_canonicalised_samples[i]

            agent_mol = redefine_bonds(mol=base_mol, adj_mat=agent_adj_mat)
            prior_mol = redefine_bonds(mol=base_mol, adj_mat=prior_adj_mat)
            baseline_mol = redefine_bonds(mol=_baseline_mol, adj_mat=baseline_adj_mat)

            # best_reward = None
            # best_valid = 0.0

            agent_score = float(self.score_fn(agent_mol))
            prior_score = float(self.score_fn(prior_mol))
            baseline_score = float(self.score_fn(baseline_mol))

            agent_valid_value = 1.0 if is_valid_mol(agent_mol) else 0.0
            prior_valid_value = 1.0 if is_valid_mol(prior_mol) else 0.0
            baseline_valid_value = 1.0 if is_valid_mol(baseline_mol) else 0.0

            agent_scores.append(agent_score)
            prior_scores.append(prior_score)
            baseline_scores.append(baseline_score)

            agent_valid_flags.append(agent_valid_value)
            prior_valid_flags.append(prior_valid_value)
            baseline_valid_flags.append(baseline_valid_value)

            # # for _ in range(self.n_samples_per_mol):
            # #     sampled_mol, _, _ = redefine_bonds_sampled(
            # #         mol=base_mol,
            # #         adj_mat=agent_adj_mat,
            # #         temperature=self.temperature,
            # #     )
            # #
            # #     reward_value = float(self.score_fn(sampled_mol))
            # #     valid_value = 1.0 if is_valid_mol(sampled_mol) else 0.0
            # #
            # #     if best_reward is None or reward_value > best_reward:
            # #         best_reward = reward_value
            # #         best_valid = valid_value
            #
            # rewards.append(best_reward if best_reward is not None else -1.0)
            # valid_flags.append(best_valid)

        agent_scores_t = torch.tensor(agent_scores, dtype=torch.float32)
        prior_scores_t = torch.tensor(prior_scores, dtype=torch.float32)
        baseline_scores_t = torch.tensor(baseline_scores, dtype=torch.float32)

        agent_valid_t = torch.tensor(agent_valid_flags, dtype=torch.float32)
        prior_valid_t = torch.tensor(prior_valid_flags, dtype=torch.float32)
        baseline_valid_t = torch.tensor(baseline_valid_flags, dtype=torch.float32)

        f_agent_score = agent_scores_t.mean().item()
        f_prior_score = prior_scores_t.mean().item()
        f_baseline_score = baseline_scores_t.mean().item()

        # eval_improv = f_agent_score - f_prior_score
        baseline_imporv = f_agent_score - f_baseline_score


        return {
            "eval_agent_scores_mean": float(agent_scores_t.mean().item()),
            "eval_baseline_scores_mean": float(baseline_scores_t.mean().item()),
            "eval_agent_valid_rate": float(agent_valid_t.mean().item()),
            "eval_baseline_valid_rate": float(baseline_valid_t.mean().item()),
            "eval_improve_mean": baseline_imporv
            # "eval_reward_std": float(rewards_t.std(unbiased=False).item()),
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
    ) -> None:
        # best_eval_improv = float("-inf")
        best_eval_improv = 0
        best_checkpoint_path = f"{save_dir}/{best_checkpoint_name}"
        latest_checkpoint_path = f"{save_dir}/latest_checkpoint.pt"

        for epoch in range(1, n_epochs + 1):
            x, h, node_mask, edge_mask = edm_sampler_fn(edm_batch_size)
            train_stats = self.train_step(x, h, node_mask, edge_mask)

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
                    f"agent_score_mean={eval_stats['eval_agent_scores_mean']:.4f} "
                    f"baseline_score_mean={eval_stats['eval_baseline_scores_mean']:.4f}\n"
                    f"                 "
                    f"eval_agent_valid_rate={eval_stats['eval_agent_valid_rate']:.4f} "
                    f"eval_baseline_valid_rate={eval_stats['eval_baseline_valid_rate']:.4f}\n"
                    f"                 "
                    f"score_improv={eval_stats['eval_improve_mean']:.4f} "
                    f"valid_rate_improv={(eval_stats['eval_agent_valid_rate'] - eval_stats['eval_baseline_valid_rate']):.4f} "
                )

                if eval_stats["eval_improve_mean"] > best_eval_improv:
                    best_eval_improv = eval_stats["eval_improve_mean"]
                    self.save_agent_head(best_checkpoint_path)
                    print("                 saved new best agent head")

        self.save_agent_head(latest_checkpoint_path)
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
    :param mol:
    :param adj_mat:
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


