import random
import re
import streamlit as st
from stspeck import speck
from utils import prepare_speck_model


import requests

import base64

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDistGeom


def generate_mock_results():
    mock_mols = Chem.SDMolSupplier('./example_structures/example.sdf')
    results = []

    for mol in mock_mols:
        results.append({'mol_block': Chem.MolToMolBlock(mol),
                       'shape_tanimoto': random.uniform(0, 1),
                       'chemical_tanimoto': random.uniform(0, 1)})
    return results


def draw_compound_image(compound: Chem.Mol):
    """
    Renders an image for a compound with labelled atoms
    :param compound: RDkit mol object
    :return: path to the generated image
    """

    pattern = re.compile("<\?xml.*\?>")
    # Create a drawer object
    d2d = Draw.rdMolDraw2D.MolDraw2DSVG(90, 90)
    # Specify the drawing options
    dopts = d2d.drawOptions()
    dopts.useBWAtomPalette()
    # dopts.addAtomIndices = True
    # Generate and save an image

    d2d.DrawMolecule(compound)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText().replace('svg:', '')
    svg = re.sub(pattern, '', svg)
    svg = "<div>" + svg + "</div>"
    return svg


def render_error():
    st.markdown(
        "<h2 style='text-align: center;'>Oops. Something went wrong.</h1>",
        unsafe_allow_html=True,
    )
    return None


def display_search_results(mols: list[dict], c_key: str = "results", cards_per_row: int = 3):
    """
    :param mols:
    :param с_key:
    :param cards_per_row:
    :return:
    """

    with st.container(height=600, key=c_key, border=False):
        for n_row, mol in enumerate(mols):
            i = n_row % cards_per_row
            if i == 0:
                cols = st.columns(cards_per_row, gap="large")
                # draw the card
            with cols[n_row % cards_per_row]:
                r_mol = Chem.MolFromMolBlock(mol["mol_block"])
                fl_mol = Chem.MolFromSmiles(Chem.MolToSmiles(r_mol))
                svg_string = draw_compound_image(fl_mol)

                st.caption(f"Shape Similarity -  {round(float(mol['shape_tanimoto']), 2)}")
                st.write(svg_string, unsafe_allow_html=True)

    return None


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
        generate_samples = st.button("Generate", type="primary")
        # view_ref = st.toggle(label="Display Reference Structure", value=True)
        # hydrogens = st.toggle(label="Display Hydrogens", value=True)

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
        # viewer_controls = st.container(height=180)

    with viewer_container:
        if view_ref:
            mol = Chem.MolFromSmiles("C1=CC(=CC=C1C(=O)O)N")

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

with output_column:
    download_sdf = st.download_button('Download', data="")
    results, ruler = st.columns([4, 1])
    mols = generate_mock_results()
    display_search_results(mols)
