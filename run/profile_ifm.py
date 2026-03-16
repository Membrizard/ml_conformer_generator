"""
IFM Inference Profiling Pipeline.

Decomposes the inertial_fragment_matching() and ff_inertial_fragment_matching()
pipelines into individually timed phases, runs multiple iterations, and writes a report.

Usage:
    python run/profile_ifm.py
    python run/profile_ifm.py --mode ff --n_samples 5 --n_iterations 3
    python run/profile_ifm.py --mode auto --device cuda
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch.nn.utils.rnn import pad_sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from src.mlconfgen import MLConformerGenerator
from src.mlconfgen.utils import (
    align_mol_to_principal_frame,
    extract_fragment,
    get_context_shape,
    ifm_get_xh_from_fragment,
    ifm_prepare_fragments_for_merge,
    ifm_prepare_gen_fragment_context,
    inverse_coord_transform,
    prepare_edm_input,
    samples_to_rdkit_mol,
    set_conformer_positions,
    split_molecule_size_constrained,
)
from src.mlconfgen.cheminformatics.shape_similarity import best_pi_rotation_by_tanimoto

WEIGHTS_DIR = PROJECT_ROOT / "src" / "mlconfgen"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile the IFM inference pipeline."
    )
    parser.add_argument(
        "--mol_path",
        type=str,
        default=str(PROJECT_ROOT / "assets" / "demo_files" / "paba.mol"),
        help="Path to input molecule (.mol/.sdf)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "ff"],
        default="auto",
        help="IFM mode: 'auto' (auto-split) or 'ff' (fixed-fragment)",
    )
    parser.add_argument(
        "--n_samples", type=int, default=10, help="Number of samples per iteration"
    )
    parser.add_argument(
        "--n_iterations", type=int, default=5, help="Number of profiling iterations"
    )
    parser.add_argument(
        "--diffusion_steps", type=int, default=100, help="EDM denoising steps"
    )
    parser.add_argument(
        "--resample_steps", type=int, default=1, help="Resample steps"
    )
    parser.add_argument(
        "--diffusion_steps_merging",
        type=int,
        default=10,
        help="Diffusion steps for merging (~10%% of diffusion_steps)",
    )
    parser.add_argument(
        "--blend_power", type=int, default=3, help="Blend power for FF merge"
    )
    parser.add_argument(
        "--variance", type=int, default=2, help="Atom count variance for FF mode"
    )
    parser.add_argument(
        "--min_frag_size", type=int, default=6, help="Minimum fragment size"
    )
    parser.add_argument(
        "--max_frag_size", type=int, default=20, help="Maximum fragment size"
    )
    parser.add_argument(
        "--max_n_atoms_final", type=int, default=None,
        help="Max atoms in final molecule (auto mode, default=ref atom count)",
    )
    parser.add_argument(
        "--min_n_atoms_final", type=int, default=None,
        help="Min atoms in final molecule (auto mode, default=ref atom count)",
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
    return parser.parse_args()


def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def align_coord(ref_coord, cand_coord):
    """Align candidate coords to reference via principal frame + pi-rotation search."""
    virtual_com = torch.mean(cand_coord, dim=0)
    ref_coord = ref_coord - virtual_com
    _, aligned_coord, _ = get_context_shape(cand_coord, include_rotation=True)
    best_coord, _ = best_pi_rotation_by_tanimoto(ref_coord, aligned_coord)
    return best_coord


def concat_masked_and_pad(xs, masks, pad_to, pad_value=0.0):
    """Concat unmasked atoms across fragments and pad to target size."""
    X = torch.stack(xs, dim=1)
    M = torch.stack(masks, dim=1)
    if M.dim() == X.dim():
        M = M.squeeze(-1)
    M = M.bool()

    B, K, N = M.shape
    feat_shape = X.shape[3:]
    flat_len = K * N

    Xf = X.reshape(B, flat_len, *feat_shape)
    Mf = M.reshape(B, flat_len)

    selected = [Xf[b][Mf[b]] for b in range(B)]
    empty = X.new_zeros((0, *feat_shape))
    selected = [t if t.numel() else empty for t in selected]

    out = pad_sequence(selected, batch_first=True, padding_value=pad_value)

    L = pad_to
    if out.size(1) < L:
        pad = out.new_full((B, L - out.size(1), *feat_shape), pad_value)
        out = torch.cat([out, pad], dim=1)
    else:
        out = out[:, :L]
    return out


# ---------------------------------------------------------------------------
# Auto-split IFM profiling
# ---------------------------------------------------------------------------
@torch.no_grad()
def profile_auto_ifm(generator, ref_mol, args, device):
    timings = {}

    # --- Phase 1: Align reference ---
    sync_device(device)
    t0 = time.perf_counter()

    ref_mol = Chem.RemoveHs(ref_mol)
    ref_context, shift, rotation, aligned_ref_coord = align_mol_to_principal_frame(ref_mol)
    aligned_ref_mol = set_conformer_positions(Chem.RWMol(ref_mol), aligned_ref_coord)

    sync_device(device)
    timings["1_align_reference"] = time.perf_counter() - t0

    n_atoms_ref = aligned_ref_coord.size(0)
    max_n_atoms_final = args.max_n_atoms_final or n_atoms_ref
    min_n_atoms_final = args.min_n_atoms_final or n_atoms_ref
    n_samples = args.n_samples
    context_norms = generator.context_norms

    # --- Phase 2: Split molecule ---
    sync_device(device)
    t0 = time.perf_counter()

    fragment_sets = split_molecule_size_constrained(
        mol=aligned_ref_mol,
        min_size=args.min_frag_size,
        max_size=args.max_frag_size,
    )
    extracted_frags = [extract_fragment(aligned_ref_mol, fs) for fs in fragment_sets]

    sync_device(device)
    timings["2_split_molecule"] = time.perf_counter() - t0

    # --- Phase 3: Align fragments + prepare EDM inputs ---
    sync_device(device)
    t0 = time.perf_counter()

    max_n_nodes = max(f.GetNumHeavyAtoms() for f in extracted_frags)

    fragment_shifts = []
    fragment_rotations = []
    ref_fragment_coords = []
    edm_inputs = []

    for frag in extracted_frags:
        n_atoms = frag.GetNumHeavyAtoms()
        f_context, f_shift, f_rotation, f_coord = align_mol_to_principal_frame(frag)
        fragment_shifts.append(f_shift)
        fragment_rotations.append(f_rotation)
        ref_fragment_coords.append(f_coord)

        node_mask, edge_mask, batch_context = prepare_edm_input(
            n_samples=n_samples,
            reference_context=f_context,
            context_norms=context_norms,
            min_n_nodes=n_atoms,
            max_n_nodes=n_atoms,
            device=device,
            pad_to=max_n_nodes,
        )
        edm_inputs.append({
            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "batch_context": batch_context,
        })

    # Build total masks
    total_node_mask = torch.cat([x["node_mask"] for x in edm_inputs], 0)
    total_batched_context = torch.cat([x["batch_context"] for x in edm_inputs])

    helper_node_mask = total_node_mask.clone().squeeze()
    batch_size = helper_node_mask.size(0)
    mn = helper_node_mask.size(1)
    total_edge_mask = helper_node_mask.unsqueeze(1) * helper_node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(mn, dtype=torch.bool, device=device).unsqueeze(0)
    total_edge_mask *= diag_mask
    total_edge_mask = total_edge_mask.view(batch_size * mn * mn, 1)

    sync_device(device)
    timings["3_align_frags_prepare_input"] = time.perf_counter() - t0

    # --- Phase 4: EDM fragment generation ---
    sync_device(device)
    t0 = time.perf_counter()

    total_x, total_h = generator.generative_model(
        total_node_mask,
        total_edge_mask,
        total_batched_context,
        resample_steps=args.resample_steps,
    )

    sync_device(device)
    timings["4_edm_fragment_generation"] = time.perf_counter() - t0

    # --- Phase 5: Align generated fragments + inverse transforms ---
    sync_device(device)
    t0 = time.perf_counter()

    x_fragments = torch.split(total_x, n_samples, dim=0)
    h_fragments = torch.split(total_h, n_samples, dim=0)
    node_masks = torch.split(total_node_mask, n_samples, 0)

    coord_for_merge = []
    for i, frag_coord in enumerate(x_fragments):
        batch_shift = fragment_shifts[i].unsqueeze(0).expand(n_samples, -1).to(device)
        batch_rot = fragment_rotations[i].transpose(0, 1).unsqueeze(0).expand(n_samples, 3, -1).to(device)

        aligned_x = []
        frag_coord_cpu = frag_coord.to("cpu")
        for old_x in frag_coord_cpu:
            aligned_x.append(align_coord(ref_fragment_coords[i], old_x))

        aligned_x = torch.stack(aligned_x, dim=0).to(device)
        new_x = inverse_coord_transform(
            coord=aligned_x, shift=batch_shift, rotation=batch_rot.transpose(1, 2)
        )
        coord_for_merge.append(new_x)

    sync_device(device)
    timings["5_align_and_transform"] = time.perf_counter() - t0

    # --- Phase 6: Assemble z_seed ---
    sync_device(device)
    t0 = time.perf_counter()

    merged_x = concat_masked_and_pad(coord_for_merge, node_masks, pad_to=max_n_atoms_final)
    merged_h = concat_masked_and_pad(h_fragments, node_masks, pad_to=max_n_atoms_final)
    z_seed = torch.cat([merged_x, merged_h], dim=2).to(device)

    sync_device(device)
    timings["6_assemble_z_seed"] = time.perf_counter() - t0

    # --- Phase 7: Merge fragments (partial denoising) ---
    sync_device(device)
    t0 = time.perf_counter()

    merging_node_mask, merging_edge_mask, batch_ref_context = prepare_edm_input(
        n_samples=n_samples,
        reference_context=ref_context,
        context_norms=context_norms,
        min_n_nodes=min_n_atoms_final,
        max_n_nodes=max_n_atoms_final,
        device=device,
    )

    edm = generator.generative_model
    n_s, n_nd, _ = merging_node_mask.size()
    z = z_seed

    for s in edm.time_steps:
        if s > args.diffusion_steps_merging:
            continue
        s_array = torch.full([n_s, 1], fill_value=s, device=z.device)
        t_array = s_array + 1.0
        s_array = s_array / edm.T
        t_array = t_array / edm.T
        for _ in range(args.resample_steps):
            z = edm.sample_p_zs_given_zt(
                s_array, t_array, z, merging_node_mask, merging_edge_mask, batch_ref_context
            )
        z = edm.sample_p_zs_given_zt(
            s_array, t_array, z, merging_node_mask, merging_edge_mask, batch_ref_context
        )
    final_x, final_h = edm.sample_p_xh_given_z0(
        z, merging_node_mask, merging_edge_mask, batch_ref_context
    )

    sync_device(device)
    timings["7_merge_fragments"] = time.perf_counter() - t0

    # --- Phase 8: Samples to mol ---
    sync_device(device)
    t0 = time.perf_counter()

    mols = samples_to_rdkit_mol(
        positions=final_x, one_hot=final_h,
        node_mask=merging_node_mask, atom_decoder=generator.atom_decoder,
    )

    sync_device(device)
    timings["8_samples_to_mol"] = time.perf_counter() - t0

    timings["total"] = sum(timings.values())
    timings["n_valid"] = len(mols)
    timings["n_fragments"] = len(fragment_sets)
    return timings


# ---------------------------------------------------------------------------
# Fixed-fragment IFM profiling
# ---------------------------------------------------------------------------
@torch.no_grad()
def profile_ff_ifm(generator, merger, ref_mol, args, device):
    timings = {}

    n_samples = args.n_samples
    g_context_norms = generator.context_norms
    m_context_norms = merger.context_norms

    # --- Phase 1: Align reference ---
    sync_device(device)
    t0 = time.perf_counter()

    ref_mol = Chem.RemoveHs(ref_mol)
    ref_context, shift, rotation, aligned_ref_coord = align_mol_to_principal_frame(ref_mol)
    aligned_ref_mol = set_conformer_positions(Chem.RWMol(ref_mol), aligned_ref_coord)

    sync_device(device)
    timings["1_align_reference"] = time.perf_counter() - t0

    # --- Phase 2: Split + extract fragments ---
    sync_device(device)
    t0 = time.perf_counter()

    fragment_sets = split_molecule_size_constrained(
        mol=aligned_ref_mol,
        min_size=args.min_frag_size,
        max_size=args.max_frag_size,
    )
    assert len(fragment_sets) == 2, (
        f"FF IFM requires exactly 2 fragments, got {len(fragment_sets)}"
    )
    fixed_fragment = extract_fragment(aligned_ref_mol, fragment_sets[0])
    ref_fragment = extract_fragment(aligned_ref_mol, fragment_sets[1])

    sync_device(device)
    timings["2_split_molecule"] = time.perf_counter() - t0

    # --- Phase 3: Calculate gen fragment context ---
    sync_device(device)
    t0 = time.perf_counter()

    _, _, _, ref_fragment_coords = align_mol_to_principal_frame(ref_fragment)

    n_nodes = aligned_ref_coord.size(0)
    min_n_nodes = n_nodes - args.variance
    max_n_nodes = n_nodes + args.variance

    ff_coord = torch.tensor(
        fixed_fragment.GetConformer().GetPositions(), dtype=torch.float32
    ).to(device)

    node_mask, edge_mask, batch_context = prepare_edm_input(
        n_samples=n_samples,
        reference_context=ref_context,
        context_norms=m_context_norms,
        min_n_nodes=min_n_nodes,
        max_n_nodes=max_n_nodes,
        device=device,
    )

    ff_n_nodes = torch.sum(node_mask, dim=1).to(torch.long)

    (frag_node_mask, frag_edge_mask, batched_normed_frag_context,
     ff_shift, ff_rotation) = ifm_prepare_gen_fragment_context(
        fixed_fragment_x=ff_coord,
        reference_context=ref_context,
        context_norms=g_context_norms,
        n_nodes=ff_n_nodes,
        max_n_nodes=max_n_nodes,
        min_n_nodes=min_n_nodes,
        device=device,
    )

    sync_device(device)
    timings["3_prepare_gen_fragment_context"] = time.perf_counter() - t0

    # --- Phase 4: EDM fragment generation ---
    sync_device(device)
    t0 = time.perf_counter()

    total_x, total_h = generator.generative_model(
        frag_node_mask,
        frag_edge_mask,
        batched_normed_frag_context,
        resample_steps=args.resample_steps,
    )

    sync_device(device)
    timings["4_edm_fragment_generation"] = time.perf_counter() - t0

    # --- Phase 5: Align generated fragments ---
    sync_device(device)
    t0 = time.perf_counter()

    aligned_x = []
    frag_coord_cpu = total_x.to("cpu")
    for old_x in frag_coord_cpu:
        aligned_x.append(align_coord(ref_fragment_coords, old_x))

    aligned_x = torch.stack(aligned_x, dim=0).to(device)
    new_x = inverse_coord_transform(coord=aligned_x, shift=ff_shift, rotation=ff_rotation)

    sync_device(device)
    timings["5_align_and_transform"] = time.perf_counter() - t0

    # --- Phase 6: Prepare fragments for merge ---
    sync_device(device)
    t0 = time.perf_counter()

    fixed_fragment_x, fixed_fragment_h = ifm_get_xh_from_fragment(
        fixed_fragment=fixed_fragment, device=device
    )
    z_known, fixed_mask = ifm_prepare_fragments_for_merge(
        fixed_fragment_x=fixed_fragment_x,
        fixed_fragment_h=fixed_fragment_h,
        gen_fragments_x=new_x.to(device),
        gen_fragments_h=total_h,
        device=device,
        max_n_nodes=max_n_nodes,
    )

    sync_device(device)
    timings["6_prepare_merge"] = time.perf_counter() - t0

    # --- Phase 7: Merge with injection ---
    sync_device(device)
    t0 = time.perf_counter()

    final_x, final_h = merger.generative_model.merge_fragments_with_injection(
        node_mask,
        edge_mask,
        fixed_mask,
        context=batch_context,
        z_seed=z_known,
        diffusion_level=args.diffusion_steps_merging,
        resample_steps=args.resample_steps,
        blend_power=args.blend_power,
    )

    sync_device(device)
    timings["7_merge_with_injection"] = time.perf_counter() - t0

    # --- Phase 8: Samples to mol ---
    sync_device(device)
    t0 = time.perf_counter()

    mols = samples_to_rdkit_mol(
        positions=final_x, one_hot=final_h,
        node_mask=node_mask, atom_decoder=merger.atom_decoder,
    )

    sync_device(device)
    timings["8_samples_to_mol"] = time.perf_counter() - t0

    timings["total"] = sum(timings.values())
    timings["n_valid"] = len(mols)
    timings["n_fixed_atoms"] = fixed_fragment.GetNumHeavyAtoms()
    return timings


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
def format_report(args, timings_list, load_time, mol_name, n_heavy_atoms):
    lines = []
    lines.append("=" * 55)
    lines.append(f"IFM Profiling Report ({args.mode.upper()} mode)")
    lines.append("=" * 55)
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Molecule: {mol_name} ({n_heavy_atoms} heavy atoms)")
    lines.append(f"Device: {args.device}")
    lines.append(f"Diffusion steps: {args.diffusion_steps}")
    lines.append(f"Merging diffusion steps: {args.diffusion_steps_merging}")
    lines.append(f"Resample steps: {args.resample_steps}")
    lines.append(f"Samples per iteration: {args.n_samples}")
    lines.append(f"Fragment size range: {args.min_frag_size}-{args.max_frag_size}")
    lines.append(f"Iterations: {args.n_iterations}")
    if args.mode == "ff":
        lines.append(f"Variance: {args.variance}")
        lines.append(f"Blend power: {args.blend_power}")
    lines.append("")
    lines.append(f"Model loading time: {load_time:.4f}s")
    lines.append("")

    # Separate timing keys from metadata keys
    meta_keys = {"n_valid", "n_fragments", "n_fixed_atoms"}
    phase_keys = [k for k in timings_list[0].keys() if k not in meta_keys]
    phase_values = {k: np.array([t[k] for t in timings_list]) for k in phase_keys}

    lines.append("--- Execution Time (seconds) ---")
    lines.append(f"{'Phase':<35} {'Mean':>10} {'Std':>10}")
    for key in phase_keys:
        vals = phase_values[key]
        lines.append(f"{key:<35} {vals.mean():>10.4f} {vals.std():>10.4f}")

    lines.append("")

    valid_counts = np.array([t["n_valid"] for t in timings_list])
    lines.append(
        f"Valid samples: {valid_counts.mean():.1f} "
        f"\u00b1 {valid_counts.std():.1f} / {args.n_samples}"
    )

    if "n_fragments" in timings_list[0]:
        lines.append(f"Fragments: {timings_list[0]['n_fragments']}")
    if "n_fixed_atoms" in timings_list[0]:
        lines.append(f"Fixed fragment atoms: {timings_list[0]['n_fixed_atoms']}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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

    # --- Load models ---
    print(f"Loading models for {args.mode.upper()} mode ...")
    sync_device(device)
    t_load_start = time.perf_counter()

    # Fragment generator always uses 6-39 weights
    generator = MLConformerGenerator(
        diffusion_steps=args.diffusion_steps,
        device=device,
        edm_weights=str(WEIGHTS_DIR / "edm_moi_chembl_6_39_fragments.pt"),
        adj_mat_seer_weights=str(WEIGHTS_DIR / "adj_mat_seer_chembl_15_39.pt"),
    )

    merger = None
    if args.mode == "ff":
        # FF mode uses a separate merger with 15-39 weights
        merger = MLConformerGenerator(
            diffusion_steps=args.diffusion_steps,
            device=device,
            edm_weights=str(WEIGHTS_DIR / "edm_moi_chembl_15_39.pt"),
            adj_mat_seer_weights=str(WEIGHTS_DIR / "adj_mat_seer_chembl_15_39.pt"),
        )

    sync_device(device)
    load_time = time.perf_counter() - t_load_start
    print(f"Models loaded in {load_time:.2f}s")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # --- Run profiling iterations ---
    timings_list = []
    for i in range(args.n_iterations):
        print(f"Iteration {i + 1}/{args.n_iterations} ...", end=" ", flush=True)

        if args.mode == "auto":
            timings = profile_auto_ifm(generator, reference_mol, args, device)
        else:
            timings = profile_ff_ifm(generator, merger, reference_mol, args, device)

        timings_list.append(timings)
        print(
            f"total={timings['total']:.2f}s, "
            f"valid={timings['n_valid']}/{args.n_samples}"
        )

    # --- Generate and write report ---
    report = format_report(args, timings_list, load_time, mol_name, n_heavy_atoms)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"profile_ifm_{args.mode}_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n{report}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
