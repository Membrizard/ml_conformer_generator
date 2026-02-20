import torch
from torch.nn.utils.rnn import pad_sequence

from rdkit import Chem
from .utils import (
                    split_molecule_size_constrained,
                    extract_fragment,
                    inverse_coord_transform,
                    prepare_edm_input,
                    samples_to_rdkit_mol,
                    prepare_masks,
                    get_moment_of_inertia_tensor,
                    ifm_standardize_mol,
                    ifm_get_xh_from_fragment,
                    ifm_prepare_fragments_for_merge,
                    ifm_prepare_gen_fragment_context,
                    )
from .cheminformatics.pipeline import set_conformer_positions, rotate_coord, tanimoto_score
from openbabel import openbabel


def strip_mol(mol: Chem.Mol) -> Chem.Mol:
    rw = Chem.RWMol()
    for a in mol.GetAtoms():
        rw.AddAtom(Chem.Atom(a.GetAtomicNum()))
    conf_in = mol.GetConformer(0)
    conf_out = Chem.Conformer(mol.GetNumAtoms())
    conf_out.SetPositions(conf_in.GetPositions())
    rw.AddConformer(conf_out, assignId=True)
    for b in mol.GetBonds():
        rw.AddBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType())
    out = rw.GetMol()

    return out


def predict_bonds_openbabel(mol: Chem.Mol, optimize_geometry: bool = True) -> Chem.Mol:
    ob_conv = openbabel.OBConversion()
    ob_conv.SetInAndOutFormats("xyz", "mol")
    obmol = openbabel.OBMol()
    xyz_block = Chem.MolToXYZBlock(mol)
    ob_conv.ReadString(obmol, xyz_block)

    obmol.ConnectTheDots()
    obmol.PerceiveBondOrders()

    mol_block = ob_conv.WriteString(obmol)
    raw_mol = Chem.MolFromMolBlock(mol_block)
    if raw_mol:
        out_mol = strip_mol(raw_mol)
        out_mol = ifm_standardize_mol(mol=out_mol, optimize_geometry=optimize_geometry)
    else:
        out_mol = None

    return out_mol


def inertial_fragment_matching( ref_mol,  # Reference rdkit Mol
                                n_samples,  # number of samples
                                generator,  # MLConformerGenerator object
                                resample_steps,  # resample steps
                                diffusion_steps_merging,  # diffusion steps for merging approx 10% from model diffusion steps
                                min_frag_size,  # Minimal fragment size in number of heavy atoms
                                max_frag_size,  # Maximal fragment size in number of heavy atoms
                                max_n_atoms_final,  # Max n_atoms in the final molecule
                                min_n_atoms_final,  # Min n_atoms in the final molecule
                                max_iter,  # Max iterations for molecule splitting
                                verbose,  # Verbose flag
                                ):

    context_norms = generator.context_norms
    device = generator.device

    # Strip of Hs and align Reference to principal Inertial Frame, saving rotation and shift
    ref_mol = Chem.RemoveHs(ref_mol)
    ref_context, shift, rotation, aligned_ref_coord = align_mol(ref_mol)

    aligned_ref_mol = set_conformer_positions(ref_mol, aligned_ref_coord)

    # Split Reference molecule into fragments

    fragment_sets = split_molecule_size_constrained(
                                                    mol=aligned_ref_mol,
                                                    min_size=min_frag_size,
                                                    max_size=max_frag_size,
                                                    max_iter=max_iter,
                                                    verbose=verbose)

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
    # Prepare concatenatable edm inputs for all fragments
    for frag in extracted_frags:
        n_atoms = frag.GetNumHeavyAtoms()
        f_context, f_shift, f_rotation, f_coord = align_mol(frag)
        fragment_contexts.append(f_context)
        fragment_shifts.append(f_shift)
        fragment_rotations.append(f_rotation)
        ref_fragment_coords.append(f_coord)

        node_mask, edge_mask, batch_context = ifm_prepare_edm_input(
            n_samples=n_samples,
            reference_context=f_context,
            context_norms=context_norms,
            n_atoms=n_atoms,
            max_n_nodes=max_n_nodes,
            device=device,
        )

        edm_inputs.append({
            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "batch_context": batch_context
        })

    total_node_mask = torch.cat([x["node_mask"] for x in edm_inputs], 0)
    total_batched_context = torch.cat([x["batch_context"] for x in edm_inputs])

    # Total Edge mask is a bit trickier:
    helper_node_mask = total_node_mask.clone().squeeze()
    batch_size = helper_node_mask.size(0)
    max_n_nodes = helper_node_mask.size(1)

    # Compute Total Edge Mask
    total_edge_mask = helper_node_mask.unsqueeze(1) * helper_node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(total_edge_mask.size(1), dtype=torch.bool, device=device).unsqueeze(0)
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
        batch_shift = fragment_shifts[i].unsqueeze(0).expand(n_samples, -1).to(device)
        batch_rot = fragment_rotations[i].transpose(0, 1).unsqueeze(0).expand(n_samples, 3, -1).to(device)

        # Align generated fragments to principal frame to maximize ref fragment volume overlay
        aligned_x = []
        frag_coord = frag_coord.to('cpu')
        for old_x in frag_coord:
            aligned_x.append(align_coord(ref_fragment_coords[i], old_x))

        aligned_x = torch.stack(aligned_x, dim=0).to(device)

        new_x = inverse_coord_transform(coord=aligned_x,
                                        shift=batch_shift,
                                        rotation=batch_rot.transpose(1, 2),
                                        )

        # Save prepared coordinates for Merging the fragments
        coord_for_merge.append(new_x)

    # We merge all the fragments by dropping masked atoms to get a proper z_seed
    merged_x = concat_masked_and_pad(coord_for_merge, node_masks, pad_to=max_n_atoms_final)
    merged_h = concat_masked_and_pad(h_fragments, node_masks, pad_to=max_n_atoms_final)

    z_seed = torch.cat([merged_x, merged_h], dim=2).to(device)

    # Here we prepare masks as for normal generation
    merging_node_mask, merging_edge_mask, batch_ref_context = prepare_edm_input(
        n_samples=n_samples,
        reference_context=ref_context,
        context_norms=context_norms,
        min_n_nodes=min_n_atoms_final,
        max_n_nodes=max_n_atoms_final,
        device=device,
    )

    with torch.no_grad():
        final_x, final_h = generator.generative_model.merge_fragments(
            node_mask=merging_node_mask,
            edge_mask=merging_edge_mask,
            context=batch_ref_context,
            z_seed=z_seed,
            diffusion_level=diffusion_steps_merging,
            resample_steps=resample_steps,
        )

    mols = samples_to_rdkit_mol(
        positions=final_x, one_hot=final_h, node_mask=merging_node_mask, atom_decoder=generator.atom_decoder
    )

    return mols


def ifm_prepare_edm_input(
    n_samples: int,
    reference_context: torch.Tensor,
    context_norms: dict,
    n_atoms: int,
    max_n_nodes: int,
    device: torch.device,
):

    # Create a random list of sizes between min_n_nodes and max_n_nodes of length n_samples

    # nodesxsample = torch.randint(min_n_atoms, max_n_atoms + 1, (n_samples,))
    nodesxsample = torch.full((n_samples,), fill_value=n_atoms, dtype=torch.long)

    node_mask, edge_mask = prepare_masks(
        n_nodes=nodesxsample,
        max_n_nodes=max_n_nodes,
        device=device,
    )

    normed_context = (
        (reference_context - context_norms["mean"]) / context_norms["mad"]
    ).to(device)

    batch_context = normed_context.unsqueeze(0).repeat(n_samples, 1)

    batch_context = batch_context.unsqueeze(1).repeat(1, max_n_nodes, 1) * node_mask

    return (
        node_mask,
        edge_mask,
        batch_context,
    )


def align_coord(ref_coord, cand_coord):
    # move coord to center
    virtual_com = torch.mean(cand_coord, dim=0)
    ref_coord = ref_coord - virtual_com

    # Get Coords in Principal Frame
    ref_context, aligned_coord, rotation = ifm_get_context_shape(cand_coord)

    # Try All 3 rotations, to account for eigenvalues equivariance
    pi = torch.pi
    rotations = [
        torch.tensor([pi, 0, 0]),
        torch.tensor([0, pi, 0]),
        torch.tensor([0, 0, pi]),
    ]

    shape_tanimoto = tanimoto_score(ref_coord, aligned_coord)
    best_coord = aligned_coord

    # Calculate Best shape similarity Tanimoto score
    for angles in rotations:
        rot_coord = rotate_coord(coord=aligned_coord, angles=angles)
        score = tanimoto_score(ref_coord, rot_coord)
        if score > shape_tanimoto:
            shape_tanimoto = score
            best_coord = rot_coord

    return best_coord


def align_mol(mol):
    conf = mol.GetConformer()
    ref_coord = torch.tensor(conf.GetPositions(), dtype=torch.float32)

    # move coord to center
    virtual_com = torch.mean(ref_coord, dim=0)
    ref_coord = ref_coord - virtual_com

    shift = - virtual_com

    ref_context, aligned_coord, rotation = ifm_get_context_shape(ref_coord)

    return ref_context, shift, rotation, aligned_coord


def concat_masked_and_pad(xs, masks, pad_extra=0, pad_to=None, pad_value=0.0):
    """
    xs:    tuple/list of tensors, each (B, N, *D)
    masks: tuple/list of masks,  each (B, N, 1) or (B, N) (bool or 0/1)

    Returns: (B, L, *D), where L depends on pad_extra / pad_to.
    """
    # (B, K, N, *D)
    X = torch.stack(xs, dim=1)

    # (B, K, N, 1) or (B, K, N)
    M = torch.stack(masks, dim=1)
    if M.dim() == X.dim():  # mask has trailing singleton like (.., 1)
        M = M.squeeze(-1)
    M = M.bool()  # (B, K, N)

    B, K, N = M.shape
    feat_shape = X.shape[3:]          # (*D)
    flat_len = K * N

    # Flatten K and N -> (B, K*N, *D)
    Xf = X.reshape(B, flat_len, *feat_shape)
    Mf = M.reshape(B, flat_len)

    # Select ragged per batch: list of (Li, *D)
    selected = [Xf[b][Mf[b]] for b in range(B)]

    # If some batch has zero selected rows, ensure correct shape for padding
    empty = X.new_zeros((0, *feat_shape))
    selected = [t if t.numel() else empty for t in selected]

    # Pad to max Li in batch -> (B, Lmax, *D)
    out = pad_sequence(selected, batch_first=True, padding_value=pad_value)

    # Decide final length
    if pad_to is not None:
        L = pad_to
    else:
        L = out.size(1) + pad_extra

    # Pad/truncate to L
    if out.size(1) < L:
        pad = out.new_full((B, L - out.size(1), *feat_shape), pad_value)
        out = torch.cat([out, pad], dim=1)
    else:
        out = out[:, :L]

    return out


def ifm_get_context_shape(coord: torch.Tensor):
    """
    Finds the principal axes for the conformer,
    and calculates Moment of Inertia tensor for the conformer in principal axes.
    All atom masses are considered equal to one, to capture shape only.
    :param coord: initial coordinates of the atoms
    :return: Principal components of MOI tensor, and coordinates rotated to a principal frame as a tuple of tensors
    """
    masses = torch.ones(coord.size(0))
    moi_tensor = get_moment_of_inertia_tensor(coord, masses)
    # Diagonalize the MOI tensor using eigen decomposition
    _, eigenvectors = torch.linalg.eigh(moi_tensor)

    # Rotate points to principal axes
    rotated_points = torch.matmul(coord.to(torch.float32), eigenvectors)

    # Get the three main moments of inertia from the main diagonal
    context = torch.diag(get_moment_of_inertia_tensor(rotated_points, masses))

    return context, rotated_points, eigenvectors


# FIXED FRAGMENT GENERATION
# ------------------------------------------------------

# Function splits ref mol automatically and takes the first fragment as fixed by default.
def ff_inertial_fragment_matching(
                                    ref_mol,
                                    generator,
                                    merger,
                                    n_samples: int,
                                    variance: int,
                                    resample_steps: int,
                                    blend_power: int,
                                    merging_diffusion_level: int,
                                    min_frag_size: int,
                                    max_frag_size: int,
                                    max_iter: int = 200,
                                    verbose: bool = False,

                                  ):
    g_context_norms = generator.context_norms
    m_context_norms = merger.context_norms

    device = generator.device

    # Strip of Hs and align Reference to principal Inertial Frame, saving rotation and shift
    ref_mol = Chem.RemoveHs(ref_mol)
    ref_context, shift, rotation, aligned_ref_coord = align_mol(ref_mol)

    aligned_ref_mol = set_conformer_positions(ref_mol, aligned_ref_coord)

    # Split Reference molecule into fragments

    fragment_sets = split_molecule_size_constrained(
        mol=aligned_ref_mol,
        min_size=min_frag_size,
        max_size=max_frag_size,
        max_iter=max_iter,
        verbose=verbose)

    # Select fixed fragment
    # Automatic Hard-Coded fixed fragment selection
    # First fragment is fixed, second fragment used as a reference, only works for molecules splittable into 2 fragments

    assert len(fragment_sets) == 2

    fixed_fragment = new_mol_from_atom_indices(aligned_ref_mol, fragment_sets[0])
    ref_fragment = new_mol_from_atom_indices(aligned_ref_mol, fragment_sets[1])

    # Align ref fragment
    _, _, _, ref_fragment_coords = align_mol(ref_fragment)

    n_nodes = aligned_ref_coord.size(0)
    min_n_nodes = n_nodes - variance
    max_n_nodes = n_nodes + variance

    ff_coord = torch.tensor(fixed_fragment.GetConformer().GetPositions(), dtype=torch.float32).to(device)

    node_mask, edge_mask, batch_context = prepare_edm_input(
        n_samples=n_samples,
        reference_context=ref_context,
        context_norms=m_context_norms,
        min_n_nodes=min_n_nodes,
        max_n_nodes=max_n_nodes,
        device=device,
    )

    ff_n_nodes = torch.sum(node_mask, dim=1).to(torch.long)

    (frag_node_mask,
     frag_edge_mask,
     batched_normed_frag_context,
     shift,
     rotation) = ifm_prepare_gen_fragment_context(
        fixed_fragment_x=ff_coord,
        reference_context=ref_context,
        context_norms=g_context_norms,
        n_nodes=ff_n_nodes,
        max_n_nodes=max_n_nodes,
        min_n_nodes=min_n_nodes,
        device=device,
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
    frag_coord = total_x.to('cpu')
    for old_x in frag_coord:
        aligned_x.append(align_coord(ref_fragment_coords, old_x))

    aligned_x = torch.stack(aligned_x, dim=0).to(device)

    new_x = inverse_coord_transform(coord=aligned_x,
                                    shift=shift,
                                    rotation=rotation,
                                    )

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

    with torch.no_grad():
        final_x, final_h = merger.generative_model.merge_fragments_with_injection(
            node_mask,
            edge_mask,
            fixed_mask,
            context=batch_context,
            z_seed=z_known,
            diffusion_level=merging_diffusion_level,
            resample_steps=resample_steps,
            blend_power=blend_power,
        )

    mols = samples_to_rdkit_mol(
        positions=final_x, one_hot=final_h, node_mask=node_mask, atom_decoder=merger.atom_decoder
    )

    return mols, fixed_fragment


def new_mol_from_atom_indices(mol: Chem.Mol, atom_indices) -> Chem.Mol:
    keep = sorted(set(int(i) for i in atom_indices))

    n = mol.GetNumAtoms()
    if keep[0] < 0 or keep[-1] > n:
        raise IndexError(f"atom index out of range (0..{n - 1}): {keep}")

    keep_set = set(keep)

    # Map old atom idx -> new atom idx
    rw = Chem.RWMol()
    old2new = {}
    for old_i in keep:
        # Copy the atom object to preserve charge/isotope/aromaticity/etc.
        new_i = rw.AddAtom(Chem.Atom(mol.GetAtomWithIdx(old_i)))
        old2new[old_i] = new_i

    # Add bonds that connect kept atoms
    for b in mol.GetBonds():
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a in keep_set and c in keep_set:
            rw.AddBond(old2new[a], old2new[c], b.GetBondType())
            nb = rw.GetBondBetweenAtoms(old2new[a], old2new[c])
            nb.SetIsAromatic(b.GetIsAromatic())

    out = rw.GetMol()

    # Copy coordinates for EACH conformer
    out.RemoveAllConformers()
    for conf in mol.GetConformers():
        new_conf = Chem.Conformer(len(keep))
        new_conf.Set3D(conf.Is3D())
        for old_i in keep:
            new_conf.SetAtomPosition(old2new[old_i], conf.GetAtomPosition(old_i))
        out.AddConformer(new_conf, assignId=True)

    return out