import random
import re

import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import Draw
import streamlit.components.v1 as components

# Make colored bars using matplotlib cmap and tanimoto score


# Buttons' functions
def generate_samples_button():
    mols = generate_mock_results()
    # Save generated molecules in session
    st.session_state.generated_mols = mols

    mock_ref = Chem.MolFromSmiles("C1CC(CC(C1)N)C(=O)O")
    mock_ref = Chem.AddHs(mock_ref)
    rdDistGeom.EmbedMolecule(mock_ref, forceTol=0.001, randomSeed=12)
    # Save aligned reference molecule in session
    st.session_state.current_ref = mock_ref
    return None


def view_mol_button(mol_in_viewer):
    st.session_state.current_mol = mol_in_viewer
    st.session_state.viewer_update = True
    return None


# Working with results, rendering mol images
def generate_mock_results():
    mock_mols = Chem.SDMolSupplier("./example_structures/example.sdf", removeHs=False)
    results = []

    for mol in mock_mols:
        results.append(
            {
                "mol_block": Chem.MolToMolBlock(mol),
                "shape_tanimoto": random.uniform(0, 1),
                "chemical_tanimoto": random.uniform(0, 1),
            }
        )
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
    svg = d2d.GetDrawingText().replace("svg:", "")
    svg = re.sub(pattern, "", svg)
    svg = "<div>" + svg + "</div>"
    return svg


def display_search_results(
    mols: list[dict], c_key: str = "results", height: int = 400, cards_per_row: int = 3
):
    """
    :param mols:
    :param с_key:
    :param cards_per_row:
    :return:
    """

    def s_f(x):
        return x["shape_tanimoto"]

    mols.sort(key=s_f, reverse=True)
    with st.container(height=height, key=c_key, border=False):
        for n_row, mol in enumerate(mols):
            i = n_row % cards_per_row
            if i == 0:
                cols = st.columns(cards_per_row, gap="large")
                # draw the card
            with cols[n_row % cards_per_row]:
                r_mol = Chem.MolFromMolBlock(mol["mol_block"])
                fl_mol = Chem.MolFromSmiles(Chem.MolToSmiles(r_mol))
                svg_string = draw_compound_image(fl_mol)

                st.caption(
                    f"Shape Similarity -  {round(float(mol['shape_tanimoto']), 2)}"
                )
                st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
                st.button(
                    label="mol",
                    key=f"mol_{n_row}",
                    on_click=view_mol_button,
                    args=[r_mol],
                )
                # svg_with_tooltip()
                st.write(svg_string, unsafe_allow_html=True)

    return None


# Utility functions
def render_error():
    st.markdown(
        "<h2 style='text-align: center;'>Oops. Something went wrong.</h1>",
        unsafe_allow_html=True,
    )
    return None


# Customize Style


def apply_custom_styling():
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """

    sliders = """"
        <style>
        .stSlider [data-baseweb=slider]{
            width: 60%;
            padding: 0px 0px 0px 20px;
        }
        </style>
        """

    buttons = """
        <style>
        .element-container:has(style){
            display: none;
        }
        #button-after {
            display: none;
        }
        .element-container:has(#button-after) {
            display: none;
        }
        .element-container:has(#button-after) + div button {
            background-color: orange;
            }
        </style>
        """

    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.markdown(sliders, unsafe_allow_html=True)
    st.markdown(buttons, unsafe_allow_html=True)

    return None


def svg_with_tooltip():
    tooltip_css = """
    <style>
        .tooltip {
          position: relative;
          display: inline-block;
          border-bottom: 1px dotted black;
        }
        
        .tooltip .tooltiptext {
          visibility: hidden;
          width: 120px;
          background-color: black;
          color: #fff;
          text-align: center;
          border-radius: 6px;
          padding: 5px 0;
          
          /* Position the tooltip */
          position: absolute;
          z-index: 1;
          top: 0%;
          left: 50%;
          margin-left: -60px;
        }
        
        .tooltip:hover .tooltiptext {
          visibility: visible;
        }
    </style>
    """

    html_content = """
    <div class="tooltip">Hover over me
        <span class="tooltiptext">Tooltip text</span>
    </div>
    """

    components.html(tooltip_css + html_content)
    return None
