# TODO:re-implement IFM for ONNX


from rdkit import Chem

from ..cheminformatics.pipeline import set_conformer_positions
from ..cheminformatics.shape_similarity import best_pi_rotation_by_tanimoto
from .conformer_generator import MLConformerGeneratorONNX
from ..utils.common import apply_transform
from ..utils.mol_split import extract_fragment, split_molecule_size_constrained
from ..utils.standardizer import standardize_mol
from ..utils.config import MIN_FRAG_SIZE, MAX_FRAG_SIZE
from .utils import (align_mol_to_principal_frame_onnx,
                    concat_masked_and_pad_onnx, get_context_shape_onnx,
                    ifm_get_xh_from_fragment_onnx, ifm_prepare_fragments_for_merge_onnx,
                    ifm_prepare_gen_fragment_context_onnx, inverse_coord_transform_onnx,
                    prepare_edm_input_onnx, samples_to_rdkit_mol
                    )


def inertial_fragment_matching_onnx(
    ref_mol: Chem.Mol,  # Reference rdkit Mol
    n_samples: int,  # number of samples
    generator: MLConformerGeneratorONNX,  # MLConformerGenerator object
    merger: MLConformerGeneratorONNX = None,  # MLConformerGenerator object,
    variance: int = 1,
    resample_steps: int = 0,  # resample steps
    diffusion_steps_merging: int = 10,  # diffusion steps for merging approx 10% from model diffusion steps
    min_frag_size: int = MIN_FRAG_SIZE,  # Minimal fragment size in number of heavy atoms
    max_frag_size: int = MAX_FRAG_SIZE,  # Maximal fragment size in number of heavy atoms
    max_iter: int = 200,  # Max iterations for molecule splitting
    verbose: bool = False,  # Verbose flag
    predict_bonds: bool = False,
    optimize_geometry: bool = False,
):
    """ """

    if merger is None:
        merger = generator

    # context_norms = generator.context_norms
    g_device = generator.device
    m_device = merger.device

    g_context_norms = generator.context_norms
    m_context_norms = merger.context_norms

    # Strip of Hs and align Reference to principal Inertial Frame, saving rotation and shift
    ref_mol = Chem.RemoveHs(ref_mol)
    ref_context, shift, rotation, aligned_ref_coord = align_mol_to_principal_frame(
        ref_mol
    )

    n_nodes = aligned_ref_coord.size(0)
    min_n_atoms_final = n_nodes - variance
    max_n_atoms_final = n_nodes + variance

    aligned_ref_mol = set_conformer_positions(ref_mol, aligned_ref_coord)

    # Split Reference molecule into fragments
    fragment_sets = split_molecule_size_constrained(
        mol=aligned_ref_mol,
        min_size=min_frag_size,
        max_size=max_frag_size,
        max_iter=max_iter,
        verbose=verbose,
    )

    if len(fragment_sets) == 0:
        raise RuntimeError(
            "Could not split reference molecule into fragments, aborting IFM generation."
        )

    # Extract fragments as individual conformers
    extracted_frags = []

    for frag_set in fragment_sets:
        extracted_frags.append(extract_fragment(aligned_ref_mol, frag_set))

    fragment_contexts = []
    fragment_shifts = []
    fragment_rotations = []
    ref_fragment_coords = []

    edm_inputs = []

    # Determine the max number of nodes
    max_n_nodes = 0
    for frag in extracted_frags:
        n_atoms = frag.GetNumHeavyAtoms()
        if n_atoms > max_n_nodes:
            max_n_nodes = n_atoms

    # Align Fragments to their respective Principal Inertial Frames,
    # while remembering corresponding Shifts and Rotations
    # Prepare concatenate-able edm inputs for all fragments
    for frag in extracted_frags:
        n_atoms = frag.GetNumHeavyAtoms()
        f_context, f_shift, f_rotation, f_coord = align_mol_to_principal_frame(frag)
        fragment_contexts.append(f_context)
        fragment_shifts.append(f_shift)
        fragment_rotations.append(f_rotation)
        ref_fragment_coords.append(f_coord)

        node_mask, edge_mask, batch_context = prepare_edm_input(
            n_samples=n_samples,
            reference_context=f_context,
            context_norms=g_context_norms,
            min_n_nodes=n_atoms,
            max_n_nodes=n_atoms,
            device=g_device,
            pad_to=max_n_nodes,
        )

        edm_inputs.append(
            {
                "node_mask": node_mask,
                "edge_mask": edge_mask,
                "batch_context": batch_context,
            }
        )

    total_node_mask = torch.cat([x["node_mask"] for x in edm_inputs], 0)
    total_batched_context = torch.cat([x["batch_context"] for x in edm_inputs])

    # Total Edge mask is a bit trickier:
    helper_node_mask = total_node_mask.clone().squeeze()
    batch_size = helper_node_mask.size(0)
    max_n_nodes = helper_node_mask.size(1)

    # Compute Total Edge Mask
    total_edge_mask = helper_node_mask.unsqueeze(1) * helper_node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(
        total_edge_mask.size(1), dtype=torch.bool, device=g_device
    ).unsqueeze(0)
    total_edge_mask *= diag_mask
    total_edge_mask = total_edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1)

    # Generating samples matching all isolated fragments at the same time
    with torch.no_grad():
        total_x, total_h = generator.generative_model(
            total_node_mask,
            total_edge_mask,
            total_batched_context,
            resample_steps=resample_steps,
        )

    # Split generated tensor into samples generated per fragment
    x_fragments = torch.split(total_x, n_samples, dim=0)
    h_fragments = torch.split(total_h, n_samples, dim=0)
    node_masks = torch.split(total_node_mask, n_samples, 0)

    # Batchify shifts and rotations
    # Apply corresponding inverse coord transforms to each fragment coordinates
    coord_for_merge = []

    for i, frag_coord in enumerate(x_fragments):
        batch_shift = fragment_shifts[i].unsqueeze(0).expand(n_samples, -1).to(m_device)
        batch_rot = (
            fragment_rotations[i]
            .transpose(0, 1)
            .unsqueeze(0)
            .expand(n_samples, 3, -1)
            .to(m_device)
        )

        # Align generated fragments to principal frame to maximize ref fragment volume overlay
        aligned_x = []
        frag_coord = frag_coord.to("cpu")
        for old_x in frag_coord:
            aligned_x.append(align_coord(ref_fragment_coords[i], old_x))

        aligned_x = torch.stack(aligned_x, dim=0).to(m_device)

        new_x = inverse_coord_transform(
            coord=aligned_x,
            shift=batch_shift,
            rotation=batch_rot.transpose(1, 2),
        )

        # Save prepared coordinates for Merging the fragments
        coord_for_merge.append(new_x)

    # We merge all the fragments by dropping masked atoms to get a proper z_seed
    merged_x = concat_masked_and_pad(
        coord_for_merge, node_masks, pad_to=max_n_atoms_final
    )
    merged_h = concat_masked_and_pad(h_fragments, node_masks, pad_to=max_n_atoms_final)

    z_seed = torch.cat([merged_x, merged_h], dim=2).to(m_device)

    # Here we prepare masks as for normal generation
    merging_node_mask, merging_edge_mask, batch_ref_context = prepare_edm_input(
        n_samples=n_samples,
        reference_context=ref_context,
        context_norms=m_context_norms,
        min_n_nodes=min_n_atoms_final,
        max_n_nodes=max_n_atoms_final,
        device=m_device,
    )

    with torch.no_grad():
        final_x, final_h = merger.generative_model.ifm_merge_fragments(
            node_mask=merging_node_mask,
            edge_mask=merging_edge_mask,
            context=batch_ref_context,
            z_seed=z_seed,
            diffusion_level=diffusion_steps_merging,
            resample_steps=resample_steps,
        )

    ifm_mols = samples_to_rdkit_mol(
        positions=final_x,
        one_hot=final_h,
        node_mask=merging_node_mask,
        atom_decoder=merger.atom_decoder,
    )

    if predict_bonds:
        raw_mols = merger.predict_bonds(ifm_mols)
        ifm_mols = []
        for f_mol in raw_mols:
            std_mol = standardize_mol(
                mol=f_mol, optimize_geometry=optimize_geometry, ifm_mode=True
            )
            if std_mol:
                ifm_mols.append(std_mol)

    return ifm_mols


# FIXED FRAGMENT GENERATION
# ------------------------------------------------------


def ff_inertial_fragment_matching_onnx(
    fixed_fragment: Chem.Mol | set,
    n_samples: int,
    generator: MLConformerGeneratorONNX,
    ref_conformer: Chem.Mol = None,
    reference_context: torch.Tensor = None,
    n_atoms: int = None,
    merger: MLConformerGeneratorONNX = None,
    variance: int = 1,
    resample_steps: int = 0,
    blend_power: int = 3,
    merging_diffusion_level: int = 10,
    predict_bonds: bool = False,
    optimize_geometry: bool = False,
):
    """
    You can set Fixed Fragment as a mol object or as a set of indexes of atoms in ref_mol!!!

    """

    if merger is None:
        merger = generator

    g_context_norms = generator.context_norms
    m_context_norms = merger.context_norms

    g_device = generator.device
    m_device = merger.device

    # Strip of Hs and align Reference to principal Inertial Frame, saving rotation and shift

    if ref_conformer:
        ref_conformer = Chem.RemoveHs(ref_conformer)
        ref_context, shift, rotation, aligned_ref_coord = align_mol_to_principal_frame_onnx(
            ref_conformer
        )

        aligned_ref_mol = set_conformer_positions(ref_conformer, aligned_ref_coord)

        if isinstance(fixed_fragment, set):
            ref_idx = {atom.GetIdx() for atom in ref_conformer.GetAtoms()}
            ref_frag_idx = ref_idx - fixed_fragment
            fixed_fragment = extract_fragment(aligned_ref_mol, fixed_fragment)
            ref_fragment = extract_fragment(aligned_ref_mol, ref_frag_idx)
            # Align ref fragment
            _, _, _, ref_fragment_coords = align_mol_to_principal_frame(ref_fragment)

            def _alignment_func(a):
                return align_coord(ref_fragment_coords, a)

        elif isinstance(fixed_fragment, Chem.Mol):
            # Apply the Reference Transformation to Fixed fragment to keep consistency
            fixed_fragment = Chem.RemoveAllHs(fixed_fragment)
            ff_conf = fixed_fragment.GetConformer()
            ff_coord = torch.tensor(ff_conf.GetPositions(), dtype=torch.float32)
            ff_coord_ref_aligned = apply_transform(ff_coord, shift, rotation)
            fixed_fragment = set_conformer_positions(
                fixed_fragment, ff_coord_ref_aligned
            )

            _alignment_func = align_mol_to_principal_frame

        else:
            raise ValueError(
                f"Unsupported fixed fragment type - {type(fixed_fragment)}"
            )

        n_nodes = aligned_ref_coord.size(0)
        min_n_nodes = n_nodes - variance
        max_n_nodes = n_nodes + variance

    elif reference_context:
        if n_atoms is None:
            raise ValueError("")

        if isinstance(fixed_fragment, Chem.Mol):
            fixed_fragment = Chem.RemoveAllHs(fixed_fragment)
            _alignment_func = align_mol_to_principal_frame
        else:
            raise ValueError(
                f"Unsupported fixed fragment type for generation from arbitrary context - {type(fixed_fragment)}"
            )

        min_n_nodes = n_atoms - variance
        max_n_nodes = n_atoms + variance
        ref_context = reference_context

    else:
        raise ValueError(
            "Either a reference RDkit Mol object or context as torch.Tensor should be provided for generation."
        )

    ff_coord = torch.tensor(
        fixed_fragment.GetConformer().GetPositions(), dtype=torch.float32
    ).to(g_device)

    node_mask, edge_mask, batch_context = prepare_edm_input(
        n_samples=n_samples,
        reference_context=ref_context,
        context_norms=m_context_norms,
        min_n_nodes=min_n_nodes,
        max_n_nodes=max_n_nodes,
        device=g_device,
    )

    ff_n_nodes = torch.sum(node_mask, dim=1).to(torch.long)

    (
        frag_node_mask,
        frag_edge_mask,
        batched_normed_frag_context,
        shift,
        rotation,
    ) = ifm_prepare_gen_fragment_context(
        fixed_fragment_x=ff_coord,
        reference_context=ref_context,
        context_norms=g_context_norms,
        n_nodes=ff_n_nodes,
        max_n_nodes=max_n_nodes,
        min_n_nodes=min_n_nodes,
        device=g_device,
    )

    with torch.no_grad():
        total_x, total_h = generator.generative_model(
            frag_node_mask,
            frag_edge_mask,
            batched_normed_frag_context,
            resample_steps=resample_steps,
        )

    # Aligning generated fragments to corresponding places inside the reference molecule

    aligned_x = []
    frag_coord = total_x.to("cpu")
    for old_x in frag_coord:
        aligned_x.append(_alignment_func(old_x))

    aligned_x = torch.stack(aligned_x, dim=0).to(m_device)

    new_x = inverse_coord_transform(
        coord=aligned_x,
        shift=shift,
        rotation=rotation,
    )

    fixed_fragment_x, fixed_fragment_h = ifm_get_xh_from_fragment(
        fixed_fragment=fixed_fragment, device=m_device
    )

    z_known, fixed_mask = ifm_prepare_fragments_for_merge(
        fixed_fragment_x=fixed_fragment_x,
        fixed_fragment_h=fixed_fragment_h,
        gen_fragments_x=new_x,
        gen_fragments_h=total_h,
        device=m_device,
        max_n_nodes=max_n_nodes,
    )

    with torch.no_grad():
        final_x, final_h = merger.generative_model.ifm_merge_fragments_with_injection(
            node_mask,
            edge_mask,
            fixed_mask,
            context=batch_context,
            z_seed=z_known,
            diffusion_level=merging_diffusion_level,
            resample_steps=resample_steps,
            blend_power=blend_power,
        )

    ff_ifm_mols = samples_to_rdkit_mol(
        positions=final_x,
        one_hot=final_h,
        node_mask=node_mask,
        atom_decoder=merger.atom_decoder,
    )

    if predict_bonds:
        raw_mols = merger.predict_bonds(ff_ifm_mols)
        ff_ifm_mols = []
        for f_mol in raw_mols:
            std_mol = standardize_mol(
                mol=f_mol, optimize_geometry=optimize_geometry, ifm_mode=True
            )
            if std_mol:
                ff_ifm_mols.append(std_mol)

    return ff_ifm_mols, fixed_fragment


def align_coord(ref_coord, cand_coord):
    # move coord to center
    virtual_com = torch.mean(cand_coord, dim=0)
    ref_coord = ref_coord - virtual_com

    # Get Coords in Principal Frame
    _, aligned_coord, _ = get_context_shape(cand_coord, include_rotation=True)

    best_coord, _ = best_pi_rotation_by_tanimoto(ref_coord, aligned_coord)
    return best_coord
