from rdkit import Chem


def show_overlay_grid_animated(
    reference_mol,
    candidate_mols_ll,   # list[list[Mol]]  (frames per cell)
    n_cols=3,
    width=300,
    height=300,
    interval=80,         # ms per frame
    loop="forward",      # "forward", "backward", "backAndForth"
):
    def mols_to_xyz_frames(mols):
        # xyz "trajectory": concatenate multiple XYZ blocks
        # py3Dmol's xyz reader can treat concatenated XYZ as frames via addModelsAsFrames
        return "".join(Chem.MolToXYZBlock(m) for m in mols)

    n_cells = len(candidate_mols_ll)
    n_rows = (n_cells + n_cols - 1) // n_cols

    view = py3Dmol.view(
        viewergrid=(n_rows, n_cols),
        width=width * n_cols,
        height=height * n_rows,
    )

    ref_block = Chem.MolToXYZBlock(reference_mol)

    for i, frames in enumerate(candidate_mols_ll):
        r, c = divmod(i, n_cols)

        # --- Reference (static) as model 0
        view.addModel(ref_block, "xyz", viewer=(r, c))
        view.setStyle(
            {"model": 0},
            {"stick": {"color": "magenta", "radius": 0.05}},
            viewer=(r, c),
        )

        # --- Candidates (animated frames) as model 1
        if frames is None or len(frames) == 0:
            continue

        frames_xyz = mols_to_xyz_frames(frames)
        view.addModelsAsFrames(frames_xyz, "xyz", viewer=(r, c))
        view.setStyle(
            {"model": 1},
            {"stick": {"radius": 0.2}},
            viewer=(r, c),
        )

        # Red dot at origin (per cell)
        view.addSphere(
            {
                "center": {"x": 0, "y": 0, "z": 0},
                "radius": 0.3,
                "color": "red",
                "opacity": 1.0,
            },
            viewer=(r, c),
        )

        view.zoomTo(viewer=(r, c))

    # Animate all viewers
    view.animate({"loop": loop, "interval": interval})
    view.show()
    return None


def show_overlay_grid(
    reference_mol,
    candidate_mols,
    n_cols=3,
    width=300,
    height=300,
):

    def mol_to_block(mol):
        return Chem.MolToXYZBlock(mol)

    n_rows = (len(candidate_mols) + n_cols - 1) // n_cols

    view = py3Dmol.view(
        viewergrid=(n_rows, n_cols),
        width=width * n_cols,
        height=height * n_rows,
    )

    ref_block = mol_to_block(reference_mol)

    for i, cand in enumerate(candidate_mols):
        r = i // n_cols
        c = i % n_cols


        # Add reference (magenta)
        view.addModel(ref_block, "xyz", viewer=(r, c))
        view.setStyle(
            {"model": 0},
            {"stick": {"color": "magenta", "radius": 0.05}},
            viewer=(r, c),
        )

        # Add candidates
        cand_block = mol_to_block(cand)
        view.addModel(cand_block, "xyz", viewer=(r, c))
        view.setStyle(
                {"model":1},
                {"stick": {"radius": 0.2}},
                viewer=(r, c),
            )
        view.addSphere({
                        "center": {"x": 0, "y": 0, "z": 0},
                        "radius": 0.3,      # tweak size
                        "color": "red",
                        "opacity": 1.0
                    })

        view.zoomTo(viewer=(r, c))

    view.show()
    return None