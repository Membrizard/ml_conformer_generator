import random
import re
import streamlit as st
from stspeck import speck
from utils import (
    apply_custom_styling,
    prepare_speck_model,
    generate_samples_button,
    display_search_results,
)


import requests

import base64

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDistGeom

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

app_header = st.container(height=120)

with app_header:
    st.write("ml conformer generator")
    st.write("generate molecules...")

input_column, viewer_column, output_column = st.columns([1, 1, 1])

with input_column:
    controls = st.container(height=600)
    with controls:
        option = st.selectbox(
            "Reference Structure Examples",
            ("structure_1", "structure_2", "structure_3"),
        )

        mol_block = st.text_area("Reference Mol, XYZ or PDB block ", height=200)

        n_samples = st.slider(
            "Number of Molecules to generate",
            min_value=20,
            max_value=100,
            step=10,
            value=60,
        )

        generate_samples = st.button(
            "Generate", on_click=generate_samples_button, type="primary"
        )

with output_column:
    if st.session_state.generated_mols:
        download_sdf = st.download_button("Download", data="")
        display_search_results(st.session_state.generated_mols, height=460)


with viewer_column:
    viewer_container = st.container(height=420, border=False)
    viewer_options = st.container(height=100, border=False)

    with viewer_options:
        st.write("Viewer Options")
        ref_col, hyd_col = st.columns([1, 1])
        with ref_col:
            view_ref = st.toggle(label="Reference Structure", value=True)
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
