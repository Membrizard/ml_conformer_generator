import torch
import streamlit.components.v1 as components
import streamlit as st
from rdkit import Chem
from stspeck import speck
from utils import (
    apply_custom_styling,
    container_css,
    display_search_results,
    draw_compound_image,
    generate_samples_button,
    header_logo,
    prepare_speck_model,
    stylable_container,
)

# Initiate Model

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps:0")
else:
    device = torch.device("cpu")

# Prepare session state values
if "generated_mols" not in st.session_state:
    st.session_state.generated_mols = None

if "current_mol" not in st.session_state:
    st.session_state.current_mol = None

if "current_ref" not in st.session_state:
    st.session_state.current_ref = None

if "viewer_update" not in st.session_state:
    st.session_state.viewer_update = False

if "running" not in st.session_state:
    st.session_state.running = False

# Page setup
st.set_page_config(
    page_title="ML Conformer Generator",
    # page_icon="./frontend/assets/quantori_favicon.ico",
    layout="wide",
)

apply_custom_styling()

app_header = stylable_container(
    key="app_header",
    css_styles=container_css,
)
with app_header:
    logo_c, title_c, _ = st.columns([0.2, 1, 1])
    with logo_c:
        header_logo("./assets/mlconfgen_logo_cosmo.png")
    with title_c:
        st.title("ML Conformer Generator")
        st.write("Generate and inspect molecules based on a reference conformer")


app_container = stylable_container(
    key="app",
    css_styles=container_css,
)
with app_container:
    input_column, viewer_column, output_column = st.columns([1, 1, 1])

    with input_column:
        controls = st.container(height=680, border=False)
        with controls:
            st.header("Input")
            st.divider()

            ref_mol = None
            uploaded_mol = st.file_uploader(
                "Reference Structure: Mol or PDB file ",
                accept_multiple_files=False,
                type=["mol", "pdb"],
                disabled=st.session_state.running,
            )

            if uploaded_mol is not None:
                mol_data = uploaded_mol.getvalue().decode("utf-8")
                filename = uploaded_mol.name
                if filename.endswith(".mol"):
                    ref_mol = Chem.MolFromMolBlock(mol_data)
                elif filename.endswith(".pdb"):
                    ref_mol = Chem.MolFromPDBBlock(mol_data)
                else:
                    # TODO Process the error
                    st.write("Bad file type")

            if ref_mol:
                f_ref_mol = Chem.MolFromSmiles(Chem.MolToSmiles(ref_mol))
                input_mol_image = draw_compound_image(f_ref_mol)
                components.html(input_mol_image)

                # ref_mol = rdDetermineBonds.DetermineBonds(ref_mol)

            n_samples_slider_c, _, variance_c = st.columns([3, 1, 3])

            with n_samples_slider_c:
                n_samples = st.slider(
                    "Number of Molecules to generate",
                    min_value=10,
                    max_value=40,
                    step=5,
                    value=25,
                    disabled=st.session_state.running,
                )

                diffusion_steps = st.slider(
                    "Diffusion steps",
                    min_value=20,
                    max_value=100,
                    step=5,
                    value=100,
                    disabled=st.session_state.running,
                )
            with variance_c:
                variance = st.number_input(
                    "Variance in Number of Atoms ±",
                    min_value=0,
                    max_value=5,
                    value=2,
                    disabled=st.session_state.running,
                )

                if ref_mol:
                    generate_samples = st.button(
                                "Generate",
                                # on_click=generate_samples_button,
                                disabled=st.session_state.running,
                                # args=(ref_mol, n_samples, diffusion_steps, variance, device),
                                type="primary",
                                key="run_button",
                            )
                else:
                    generate_samples = None

    if st.session_state.running:
        with output_column:
            st.write("Hidden")

        with viewer_column:
            st.write("Hidden")

        with st.spinner("Generating ..."):
            generate_samples_button(ref_mol, n_samples, diffusion_steps, variance, device)
            st.rerun()

    else:
        with output_column:

            header_c, button_c = st.columns([2.5, 1])
            with header_c:
                st.header("Output")

            st.divider()
            if st.session_state.generated_mols:
                with button_c:
                    st.write("")
                    download_sdf = st.download_button("Download", data="")

                st.caption("Shape Similarity to Reference:")
                # if st.session_state.generated_mols:
                display_search_results(st.session_state.generated_mols, height=460)
            else:
                st.write("")

        with viewer_column:
            viewer_container = st.container(height=420, border=False)
            viewer_options = st.container(height=100, border=False)

            with viewer_options:
                if st.session_state.generated_mols:
                    st.write("Viewer Options")
                    ref_col, hyd_col = st.columns([1, 1])
                    with ref_col:
                        view_ref = st.toggle(label="Reference Structure", value=True, disabled=st.session_state.running)
                    with hyd_col:
                        hydrogens = st.toggle(label="Hydrogens", value=True, disabled=st.session_state.running)

            with viewer_container:

                if st.session_state.viewer_update:
                    c_mol_index = st.session_state.current_mol
                    mol_block = st.session_state.generated_mols[c_mol_index]
                    ref_block = st.session_state.current_ref

                    if hydrogens:
                        n_mol = Chem.MolFromMolBlock(mol_block["mol_block"], removeHs=False)
                        mol = Chem.AddHs(n_mol, addCoords=True)
                        ref = Chem.MolFromMolBlock(ref_block, removeHs=False)

                    else:
                        mol = Chem.MolFromMolBlock(mol_block["mol_block"], removeHs=True)
                        ref = Chem.MolFromMolBlock(ref_block, removeHs=True)

                        # Handle reference structure
                    if view_ref:
                        json_mol = prepare_speck_model(mol, ref)
                        res = speck(data=json_mol, height="400px", aoRes=512)

                    else:
                        json_mol = prepare_speck_model(mol)
                        res = speck(data=json_mol, height="400px", aoRes=512)

if generate_samples:
    st.session_state.running = True
    st.session_state.viewer_update = False
    st.session_state.generated_mols = False
    st.session_state.current_mol = False
    st.session_state.current_ref = False
    print("Reach")
    st.rerun()

# if st.session_state.running:
#     with st.spinner("Generating ..."):
#         generate_samples_button(ref_mol, n_samples, diffusion_steps, variance, device)
#         st.rerun()