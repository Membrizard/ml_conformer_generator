"""
MLConformerGenerator Inference Profiling Pipeline.

Decomposes the generate_conformers() pipeline into individually timed phases,
runs multiple iterations, collects memory statistics, and writes a report.

Usage:
    python run/profile_pipeline.py
    python run/profile_pipeline.py --n_samples 5 --n_iterations 3 --device cuda
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.mlconfgen import MLConformerGenerator
from src.mlconfgen.utils import (
    get_context_shape,
    prepare_adj_mat_seer_input,
    prepare_edm_input,
    redefine_bonds,
    samples_to_rdkit_mol,
    standardize_mol,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile the MLConformerGenerator inference pipeline."
    )
    parser.add_argument(
        "--mol_path",
        type=str,
        default=str(PROJECT_ROOT / "assets" / "demo_files" / "paba.mol"),
        help="Path to input molecule (.mol/.sdf)",
    )
    parser.add_argument(
        "--edm_weights",
        type=str,
        default=str(PROJECT_ROOT / "src" / "mlconfgen" / "edm_moi_chembl_15_39.pt"),
        help="Path to EDM weights",
    )
    parser.add_argument(
        "--ams_weights",
        type=str,
        default=str(
            PROJECT_ROOT / "src" / "mlconfgen" / "adj_mat_seer_chembl_15_39.pt"
        ),
        help="Path to AdjMatSeer weights",
    )
    parser.add_argument(
        "--n_samples", type=int, default=10, help="Number of samples per iteration"
    )
    parser.add_argument(
        "--n_iterations", type=int, default=5, help="Number of profiling iterations"
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=100,
        help="Number of diffusion denoising steps",
    )
    parser.add_argument(
        "--variance",
        type=int,
        default=2,
        help="Atom count variance from reference",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device (cpu or cuda)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "run"),
        help="Report output directory",
    )
    parser.add_argument(
        "--skip_memory",
        action="store_true",
        help="Skip memory profiling",
    )
    return parser.parse_args()


def sync_device(device: torch.device):
    """Synchronize CUDA device if applicable."""
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def profile_iteration(generator, reference_mol, n_samples, variance, device):
    """
    Run one full pipeline iteration with per-phase timing.

    Returns a dict of step timings (seconds) and n_valid count.
    """
    timings = {}

    # --- Phase 1: Shape extraction ---
    sync_device(device)
    t0 = time.perf_counter()

    ref_mol = Chem.RemoveHs(reference_mol)
    ref_n_atoms = ref_mol.GetNumAtoms()
    conf = ref_mol.GetConformer()
    ref_coord = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    virtual_com = torch.mean(ref_coord, dim=0)
    ref_coord = ref_coord - virtual_com
    ref_context, _, _ = get_context_shape(ref_coord, include_rotation=True)

    sync_device(device)
    timings["1_shape_extraction"] = time.perf_counter() - t0

    min_n_nodes = max(ref_n_atoms - variance, generator.min_n_nodes)
    max_n_nodes = min(ref_n_atoms + variance, generator.max_n_nodes)
    # Ensure valid range when molecule is smaller than model's min_n_nodes
    if min_n_nodes > max_n_nodes:
        max_n_nodes = min_n_nodes

    # --- Phase 2a: EDM input preparation ---
    sync_device(device)
    t0 = time.perf_counter()

    node_mask, edge_mask, batch_context = prepare_edm_input(
        n_samples=n_samples,
        reference_context=ref_context,
        context_norms=generator.context_norms,
        min_n_nodes=min_n_nodes,
        max_n_nodes=max_n_nodes,
        device=device,
    )

    sync_device(device)
    timings["2a_edm_input_prep"] = time.perf_counter() - t0

    # --- Phase 2b: EDM diffusion (decomposed) ---
    edm = generator.generative_model
    n_samples_b, n_nodes_b, _ = node_mask.size()

    # 2b-i: Initial noise sampling
    sync_device(device)
    t0 = time.perf_counter()

    z = edm.sample_combined_position_feature_noise(n_samples_b, n_nodes_b, node_mask)

    sync_device(device)
    timings["2b_i_noise_sampling"] = time.perf_counter() - t0

    # 2b-ii: Denoising loop
    sync_device(device)
    t0 = time.perf_counter()

    for s in edm.time_steps:
        s_array = torch.full([n_samples_b, 1], fill_value=s, device=z.device)
        t_array = s_array + 1.0
        s_array = s_array / edm.T
        t_array = t_array / edm.T
        z = edm.sample_p_zs_given_zt(
            s_array, t_array, z, node_mask, edge_mask, batch_context
        )

    sync_device(device)
    timings["2b_ii_denoising_loop"] = time.perf_counter() - t0

    # 2b-iii: Final sample decoding p(x, h | z_0)
    sync_device(device)
    t0 = time.perf_counter()

    x, h = edm.sample_p_xh_given_z0(z, node_mask, edge_mask, batch_context)

    sync_device(device)
    timings["2b_iii_final_decode"] = time.perf_counter() - t0

    # --- Phase 2c: Samples to mol ---
    sync_device(device)
    t0 = time.perf_counter()

    mols = samples_to_rdkit_mol(
        positions=x,
        one_hot=h,
        node_mask=node_mask,
        atom_decoder=generator.atom_decoder,
    )

    sync_device(device)
    timings["2c_samples_to_mol"] = time.perf_counter() - t0

    # --- Phase 3a: AdjMatSeer input preparation ---
    sync_device(device)
    t0 = time.perf_counter()

    el_batch, dm_batch, b_adj_mat_batch, canonicalised_samples = (
        prepare_adj_mat_seer_input(
            mols=mols,
            dimension=generator.dimension,
            device=device,
        )
    )

    sync_device(device)
    timings["3a_ams_input_prep"] = time.perf_counter() - t0

    # --- Phase 3b: AdjMatSeer forward ---
    sync_device(device)
    t0 = time.perf_counter()

    adj_mat_batch = generator.adj_mat_seer(
        elements=el_batch, dist_mat=dm_batch, adj_mat=b_adj_mat_batch
    ).cpu()

    sync_device(device)
    timings["3b_ams_forward"] = time.perf_counter() - t0

    # --- Phase 4: Postprocessing (bond redefinition + standardisation) ---
    sync_device(device)
    t0 = time.perf_counter()

    results = []
    for i, adj_mat in enumerate(adj_mat_batch):
        f_mol = redefine_bonds(canonicalised_samples[i], adj_mat)
        std_mol = standardize_mol(f_mol, optimize_geometry=True)
        if std_mol:
            results.append(std_mol)

    sync_device(device)
    timings["4_postprocessing"] = time.perf_counter() - t0

    timings["total"] = sum(timings.values())
    timings["n_valid"] = len(results)
    return timings


def run_memory_profiling(generator, ref_mol, n_samples, device):
    """Run memory profiling using existing memory_profiling module.

    Requires torchinfo and pandas: pip install torchinfo pandas
    """
    from memory_profiling.profile_model import profile_adj_mat_seer, profile_egnn

    ref_mol_nohs = Chem.RemoveHs(ref_mol)
    n_atoms = ref_mol_nohs.GetNumAtoms()

    egnn_bytes = profile_egnn(
        generator.generative_model,
        n_samples=n_samples,
        n_nodes=n_atoms,
    )

    # Need a sample mol with connectivity for AdjMatSeer profiling
    from rdkit.Chem import rdDetermineBonds

    sample_mol = Chem.RWMol(ref_mol_nohs)
    rdDetermineBonds.DetermineConnectivity(sample_mol)
    sample_mol = sample_mol.GetMol()

    ams_bytes = profile_adj_mat_seer(
        generator.adj_mat_seer,
        sample_mol=sample_mol,
        n_samples=n_samples,
    )

    memory_info = {
        "egnn_bytes": egnn_bytes,
        "ams_bytes": ams_bytes,
        "total_bytes": egnn_bytes + ams_bytes,
    }

    if device.type == "cuda":
        memory_info["peak_gpu_bytes"] = torch.cuda.max_memory_allocated(device)

    return memory_info


def format_report(args, timings_list, memory_info, mol_name, n_heavy_atoms):
    """Format the profiling report as a string."""
    lines = []
    lines.append("=" * 50)
    lines.append("MLConformerGenerator Profiling Report")
    lines.append("=" * 50)
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Molecule: {mol_name} ({n_heavy_atoms} heavy atoms)")
    lines.append(f"Device: {args.device}")
    lines.append(f"Diffusion steps: {args.diffusion_steps}")
    lines.append(f"Samples per iteration: {args.n_samples}")
    lines.append(f"Iterations: {args.n_iterations}")
    lines.append("")

    # Timing statistics
    phase_keys = [
        k for k in timings_list[0].keys() if k not in ("n_valid",)
    ]
    phase_values = {
        k: np.array([t[k] for t in timings_list]) for k in phase_keys
    }
    valid_counts = np.array([t["n_valid"] for t in timings_list])

    lines.append("--- Execution Time (seconds) ---")
    lines.append(f"{'Phase':<25} {'Mean':>10} {'Std':>10}")
    for key in phase_keys:
        vals = phase_values[key]
        lines.append(f"{key:<25} {vals.mean():>10.4f} {vals.std():>10.4f}")

    lines.append("")
    lines.append(
        f"Valid samples: {valid_counts.mean():.1f} "
        f"\u00b1 {valid_counts.std():.1f} / {args.n_samples}"
    )

    # Memory section
    if memory_info is not None:
        lines.append("")
        lines.append("--- Memory Profile ---")
        egnn_mb = memory_info["egnn_bytes"] / (1024**2)
        ams_mb = memory_info["ams_bytes"] / (1024**2)
        total_mb = memory_info["total_bytes"] / (1024**2)
        lines.append(f"{'EGNN (torchinfo):':<28} {egnn_mb:>8.2f} MB")
        lines.append(f"{'AdjMatSeer (torchinfo):':<28} {ams_mb:>8.2f} MB")
        lines.append(f"{'Total model memory:':<28} {total_mb:>8.2f} MB")
        if "peak_gpu_bytes" in memory_info:
            peak_mb = memory_info["peak_gpu_bytes"] / (1024**2)
            lines.append(f"{'Peak GPU memory:':<28} {peak_mb:>8.2f} MB")

    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    device = torch.device(args.device)
    mol_path = Path(args.mol_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading molecule from {mol_path} ...")
    reference_mol = Chem.MolFromMolFile(str(mol_path), removeHs=False)
    if reference_mol is None:
        print(f"Error: Could not read molecule from {mol_path}")
        sys.exit(1)

    mol_name = mol_path.name
    n_heavy_atoms = Chem.RemoveHs(reference_mol).GetNumAtoms()

    # --- Phase 0: Model loading ---
    print("Loading model ...")
    sync_device(device)
    t_load_start = time.perf_counter()

    generator = MLConformerGenerator(
        diffusion_steps=args.diffusion_steps,
        device=device,
        edm_weights=args.edm_weights,
        adj_mat_seer_weights=args.ams_weights,
    )

    sync_device(device)
    load_time = time.perf_counter() - t_load_start
    print(f"Model loaded in {load_time:.2f}s")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # --- Run profiling iterations ---
    timings_list = []
    for i in range(args.n_iterations):
        print(f"Iteration {i + 1}/{args.n_iterations} ...", end=" ", flush=True)
        timings = profile_iteration(
            generator, reference_mol, args.n_samples, args.variance, device
        )
        timings_list.append(timings)
        print(
            f"total={timings['total']:.2f}s, "
            f"valid={timings['n_valid']}/{args.n_samples}"
        )

    # --- Memory profiling ---
    memory_info = None
    if not args.skip_memory:
        print("Running memory profiling ...")
        try:
            memory_info = run_memory_profiling(
                generator, reference_mol, args.n_samples, device
            )
        except Exception as e:
            print(f"Memory profiling failed: {e}")

    # --- Generate and write report ---
    report = format_report(args, timings_list, memory_info, mol_name, n_heavy_atoms)

    # Prepend model loading time to report
    load_line = f"Model loading time: {load_time:.4f}s\n"
    report = report.replace(
        "--- Execution Time",
        load_line + "\n--- Execution Time",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"profile_report_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n{report}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
