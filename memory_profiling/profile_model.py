import csv
import random

import torch
from mlconfgen import MLConformerGenerator
from mlconfgen.utils.config import CONTEXT_NORMS
from mlconfgen.utils.mol_utils import prepare_adj_mat_seer_input
from rdkit import Chem
from torchinfo import summary
from tqdm import tqdm
import numpy as np


def get_total_bytes(model_stats):
    return (
        model_stats.total_input
        + model_stats.total_output_bytes
        + model_stats.total_param_bytes
    )


def profile_egnn(
    generative_model: torch.nn.Module,
    n_samples: int,
    n_nodes: int,
    dummy_ref_context: torch.Tensor = torch.tensor(
        [53.6424, 108.3042, 151.4399], dtype=torch.float32
    ),
    context_norms: dict = CONTEXT_NORMS,
    dtype: torch.dtype = torch.float32,
):
    egnn = generative_model.dynamics

    context_norms = {key: torch.tensor(value) for key, value in context_norms.items()}

    egnn_inputs = prepare_egnn_dummy_input(
        generative_model=generative_model,
        reference_context=dummy_ref_context,
        context_norms=context_norms,
        n_samples=n_samples,
        min_n_nodes=n_nodes,
        max_n_nodes=n_nodes,
        s=50,
        timesteps=100,
        dtype=dtype,
    )

    model_stats = summary(egnn, input_data=egnn_inputs, verbose=0)

    return model_stats


def prepare_egnn_dummy_input(
    generative_model: torch.nn.Module,
    reference_context: torch.Tensor,
    context_norms: dict,
    n_samples: int = 2,
    min_n_nodes: int = 16,
    max_n_nodes: int = 20,
    s: int = 50,
    timesteps: int = 100,
    dtype: torch.dtype = torch.float32,
):
    device = generative_model.dynamics.device
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
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    normed_context = (
        (reference_context - context_norms["mean"]) / context_norms["mad"]
    ).to(device)

    batch_context = normed_context.unsqueeze(0).repeat(batch_size, 1)

    batch_context = batch_context.unsqueeze(1).repeat(1, max_n_nodes, 1) * node_mask

    z = generative_model.sample_combined_position_feature_noise(
        n_samples, max_n_nodes, node_mask
    )
    s_array = torch.full([n_samples, 1], fill_value=s, device=device)
    t_array = s_array + 1.0
    t_array = t_array / timesteps

    input_data = {
        "t": t_array.to(dtype),
        "xh": z.to(dtype),
        "node_mask": node_mask.to(dtype),
        "edge_mask": edge_mask.to(dtype),
        "context": batch_context.to(dtype),
    }

    return input_data


def profile_adj_mat_seer(
    adj_mat_seer: torch.nn.Module,
    sample_mol: Chem.Mol,
    n_samples: int,
    dtype: torch.dtype = torch.float32,
):
    mols = [sample_mol] * n_samples

    device = adj_mat_seer.device
    dimension = adj_mat_seer.dimension

    elements, dist_mat, adj_mat, _ = prepare_adj_mat_seer_input(
        mols=mols,
        dimension=dimension,
        device=device,
    )

    input_data = {
        "elements": elements.to(torch.int),
        "dist_mat": dist_mat.to(dtype),
        "adj_mat": adj_mat.to(dtype),
    }

    model_stats = summary(adj_mat_seer, input_data=input_data, verbose=1)

    return model_stats


def profile_model(
    model: MLConformerGenerator,
    csv_path: str = "memory_profile.csv",
    sample_mols: str = "./mol_examples/alkanes_C15_C39.sdf",
    dtype: torch.dtype = torch.float32,
) -> None:
    model = model.half()

    reader = Chem.SDMolSupplier(sample_mols)
    mols = [x for x in reader]

    with open(csv_path, mode="a+", newline="") as csvfile:
        fieldnames = [
            "n_samples",
            "n_atoms",
            "egnn_memory_bytes",
            "adj_mat_seer_memory_bytes",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for n_samples in tqdm(range(20, 120, 20)):
            for i, n_atoms in enumerate(range(15, 40)):
                eggn_stats = profile_egnn(
                    model.generative_model,
                    n_samples=n_samples,
                    n_nodes=n_atoms,
                    dtype=dtype,
                )
                adj_mat_seer_stats = profile_adj_mat_seer(
                    model.adj_mat_seer,
                    sample_mol=mols[i],
                    n_samples=n_samples,
                    dtype=dtype,
                )
                egnn_total_bytes = get_total_bytes(eggn_stats)
                adj_mat_seer_bytes = get_total_bytes(adj_mat_seer_stats)
                writer.writerow(
                    {
                        "n_samples": n_samples,
                        "n_atoms": n_atoms,
                        "egnn_memory_bytes": egnn_total_bytes,
                        "adj_mat_seer_memory_bytes": adj_mat_seer_bytes,
                    }
                )


def fit_profile(profile_csv: str):
    # Load CSV (assumes columns: x, y)
    df = pd.read_csv("./model_profiling/full_memory_profile_float16.csv")

    n_samples = df['n_samples'].to_numpy()
    n_atoms = df['n_atoms'].to_numpy()
    egnn_memory = df['egnn_memory_bytes'].to_numpy()
    gcn_memory = df['adj_mat_seer_memory_bytes'].to_numpy()

    y = (egnn_memory + gcn_memory) / 1024 ** 2

    # GCN is linear vs n_samles * n_atoms^2
    x = n_samples * np.power(n_atoms, 2)

    # Do least-squares fit: y ≈ m*x + b
    m, b = np.polyfit(x, y, 1)  # degree=1 → linear

    print(f"Slope (m): {m}")
    print(f"Intercept (b): {b}")

    # Predict and plot
    y_pred = m * x + b

    plt.scatter(x, y, label="Data")
    plt.plot(x, y_pred, color='red', label=f"Fit: y={m:.3f}x+{b:.3f}")
    plt.legend()
    plt.show()

    return None