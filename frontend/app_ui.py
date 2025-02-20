import random
import re
import base64
import streamlit as st
import streamlit.components.v1 as components

from stspeck import speck
from utils import (
    apply_custom_styling,
    prepare_speck_model,
    generate_samples_button,
    display_search_results,
    header_image,
    stylable_container,
)
container_css = """
            {
                background-color: #0f1116;
                border: 1.5px solid rgba(49, 51, 63);
                border-radius: 2rem;
                padding: calc(1em - 1px);
            }
            """
# Prepare session state values
if "generated_mols" not in st.session_state:
    st.session_state.generated_mols = None

if "current_mol" not in st.session_state:
    st.session_state.current_mol = None

if "current_ref" not in st.session_state:
    st.session_state.current_ref = None

if "viewer_update" not in st.session_state:
    st.session_state.viewer_update = False

# Page setup
st.set_page_config(
    page_title="ML Conformer Generator",
    # page_icon="./frontend/assets/quantori_favicon.ico",
    layout="wide",
)

apply_custom_styling()

# app_header = st.container(height=120)
app_header = stylable_container(
    key="app_header",
    css_styles=container_css,
)
with app_header:
    title_c, img_c = st.columns([1, 1])
    with title_c:
        st.write("ml conformer generator")
        st.write("generate molecules...")
    with img_c:
        header_image("./assets/header_background.png")

# app_container = st.container(height=None, border=True, )
app_container = stylable_container(
    key="app",
    css_styles=container_css,
)
with app_container:
    input_column, viewer_column, output_column = st.columns([1, 1, 1])

    with input_column:
        controls = st.container(height=630, border=False)
        with controls:
            st.header("Input")
            st.divider()
            option = st.selectbox(
                "Reference Structure Examples",
                ("structure_1", "structure_2", "structure_3"),
            )

            mol_block = st.text_area(
                "Reference Structure: Mol, XYZ or PDB block ", height=200
            )
            n_samples_slider_c, _, variance_c = st.columns([3, 1, 3])

            with n_samples_slider_c:
                n_samples = st.slider(
                    "Number of Molecules to generate",
                    min_value=20,
                    max_value=60,
                    step=10,
                    value=40,
                )
            with variance_c:
                variance = st.number_input(
                    "Variance in Number of Atoms ±", min_value=0, max_value=5, value=2
                )

            _, generate_button_c = st.columns([1.4, 1])
            with generate_button_c:
                generate_samples = st.button(
                    "Generate", on_click=generate_samples_button, type="primary"
                )

    with output_column:
        header_c, button_c = st.columns([2.5, 1])
        with header_c:
            st.header("Output")
        with button_c:
            st.write("")
            download_sdf = st.download_button("Download", data="")
            # with header_c:
            #     st.header("Output")
            # with button_c:
            #     download_sdf = st.download_button("Download", data="")
        st.divider()
        st.caption("Shape Similarity to Reference:")
        if st.session_state.generated_mols:
            display_search_results(st.session_state.generated_mols, height=460)

    with viewer_column:
        viewer_container = st.container(height=420, border=False)
        viewer_options = st.container(height=100, border=False)

        with viewer_options:
            st.write("Viewer Options")
            ref_col, hyd_col = st.columns([1, 1])
            with ref_col:
                view_ref = st.toggle(label="Reference Structure", value=False)
            with hyd_col:
                hydrogens = st.toggle(label="Hydrogens", value=True)

        with viewer_container:
            if st.session_state.viewer_update:
                mol = st.session_state.current_mol
                ref = st.session_state.current_ref

                # # Handle Hydrogens
                # if hydrogens:
                #     mol = Chem.AddHs(st.session_state.current_mol)
                #     ref = Chem.AddHs(st.session_state.current_ref)
                # else:
                #     mol = Chem.RemoveHs(st.session_state.current_mol)
                #     ref = Chem.RemoveHs(st.session_state.current_ref)

                # Handle reference structure
                if view_ref:
                    json_mol = prepare_speck_model(mol, ref)
                    res = speck(data=json_mol, height="400px", aoRes=512)

                else:
                    json_mol = prepare_speck_model(mol)
                    res = speck(data=json_mol, height="400px", aoRes=512)

with st.expander("Description of the Model"):
        st.caption("Description of the Model...")