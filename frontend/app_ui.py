import streamlit as st
from stspeck import speck
from utils import prepare_speck_model


import requests

from streamlit_ketcher import st_ketcher

import base64

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDistGeom


# Page setup
st.set_page_config(
    page_title="ML Conformer Generator",
    page_icon="./frontend/assets/quantori_favicon.ico",
    layout="wide",
)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# st.markdown(
#     """
#     <style>
#     .stSlider [data-baseweb=slider]{
#         width: 60%;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

app_header = st.container(height=120)
with app_header:
    st.write("ml conformer generator")
    st.write("generate molecules...")

input_container, output_container = st.columns([1, 1.8])


with input_container:
    controls = st.container(height=600)
    with controls:
        option = st.selectbox(
            "Reference Structure Examples",
            ("structure_1", "structure_2", "structure_3"),
        )

        mol_block = st.text_area("Reference Mol, XYZ or PDB block ", height=200)

        n_samples = st.slider("Number of Molecules to generate", min_value=20, max_value=100, step=10, value=60)

with output_container:
    viewer, viewer_controls = st.columns([2, 1])
    with viewer:
        viewer_container = st.container(height=400)

        with viewer_controls:
            view_ref = st.toggle(label="Display Reference Structure", value=True)
            hydrogens = st.toggle(label="Display Hydrogens", value=True)
            
        with viewer_container:
            if view_ref:
                mol = Chem.MolFromSmiles("C1=CC(=CC=C1C(=O)O)N")
                mol = Chem.AddHs(mol)
                rdDistGeom.EmbedMolecule(mol, forceTol=0.001, randomSeed=12)

                ref = Chem.MolFromSmiles("C1CC(CC(C1)N)C(=O)O")
                ref = Chem.AddHs(ref)
                rdDistGeom.EmbedMolecule(ref, forceTol=0.001, randomSeed=12)

                json_mol = prepare_speck_model(mol, ref)
                res = speck(data=json_mol, height="400px", aoRes=512)

            else:
                mol = Chem.MolFromSmiles("C1=CC(=CC=C1C(=O)O)N")
                mol = Chem.AddHs(mol)
                rdDistGeom.EmbedMolecule(mol, forceTol=0.001, randomSeed=12)

                # ref = Chem.MolFromSmiles('C1CC(CC(C1)N)C(=O)O')
                # ref = Chem.AddHs(ref)
                # rdDistGeom.EmbedMolecule(ref, forceTol=0.001, randomSeed=12)

                json_mol = prepare_speck_model(mol)
                res = speck(data=json_mol, height="400px", aoRes=512)
