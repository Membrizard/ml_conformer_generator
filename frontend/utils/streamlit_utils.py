import random
import re
import base64

import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import Draw
import streamlit.components.v1 as components
import matplotlib
import json

CMAP = matplotlib.cm.get_cmap("viridis")

container_css = """
            {
                background-color: #0f1116;
                border: 1.5px solid rgba(49, 51, 63);
                border-radius: 2rem;
                padding: calc(1em - 1px);
            }
            """

# Make colored bars using matplotlib cmap and tanimoto score
SVG_PALETTE = {
    1: (0.830, 0.830, 0.830),  # H
    6: (0.830, 0.830, 0.830),  # C
    7: (0.200, 0.600, 0.973),  # N
    8: (1.000, 0.400, 0.400),  # O
    9: (0.000, 0.800, 0.267),  # F
    15: (1.000, 0.502, 0.000),  # P
    16: (1.000, 1.000, 0.188),  # S
    17: (0.750, 1.000, 0.000),  # Cl
    35: (0.902, 0.361, 0.000),  # Br
}


# Functions for buttons
def generate_samples_button():
    ref, mols = generate_mock_results()
    # Save generated molecules in session
    st.session_state.generated_mols = mols

    # mock_ref = Chem.MolFromSmiles("C1CC(CC(C1)N)C(=O)O")
    # mock_ref = Chem.AddHs(mock_ref)
    # rdDistGeom.EmbedMolecule(mock_ref, forceTol=0.001, randomSeed=12)
    # Save aligned reference molecule in session
    st.session_state.current_ref = ref
    return None


def view_mol_button(mol_index):
    st.session_state.current_mol = mol_index
    st.session_state.viewer_update = True
    return None


# Working with results, rendering mol images
def generate_mock_results():
    with open("./generation_examples/generation_example_6.json") as json_file:
        data = json.load(json_file)

        def s_f(x):
            return x["shape_tanimoto"]

        samples = data["generated_molecules"]

        samples.sort(key=s_f, reverse=True)
        ref = data["aligned_reference"]

    return ref, samples


def draw_compound_image(compound: Chem.Mol):
    """
    Renders an image for a compound with labelled atoms
    :param compound: RDkit mol object
    :return: path to the generated image
    """

    pattern = re.compile("<\?xml.*\?>")
    # Create a drawer object
    d2d = Draw.rdMolDraw2D.MolDraw2DSVG(160, 160)
    # Specify the drawing options
    dopts = d2d.drawOptions()
    dopts.setAtomPalette(SVG_PALETTE)
    dopts.bondLineWidth = 1
    dopts.bondColor = (0, 0, 0)
    dopts.clearBackground = False
    # Generate and save an image

    d2d.DrawMolecule(compound)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText().replace("svg:", "")
    svg = re.sub(pattern, "", svg)
    svg = "<div>" + svg + "</div>"
    return svg


def display_search_results(
    mols: list[dict],
    c_key: str = "results",
    height: int = 400,
    cards_per_row: int = 2,
):
    """
    :param mols:
    :param с_key:
    :param cards_per_row:
    :return:
    """

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

                create_view_molecule_button(n_row, float(mol["shape_tanimoto"]), n_row)

                # st.button(
                #     label="mol",
                #     key=f"mol_{n_row}",
                #     on_click=view_mol_button,
                #     args=[r_mol],
                # )
                components.html(svg_string)
                st.divider()

    return None


def create_view_molecule_button(r_mol, score, key):
    score = round(score, 2)
    color = tuple(round(x * 255, 2) for x in CMAP(score))

    if score > 0.3:
        l_color = "#262730"
    else:
        l_color = "#d3d3d3"

    rgb_string = f"rgb{str(color[:-1])}"
    with stylable_container(
        key=f"molecule_button_{key}",
        css_styles="""
                button {"""
        + f"\nbackground-color: {rgb_string};\n"
        + f"\ncolor: {l_color};\n"
        + """border-radius: 2px;
                    width: 100%;
                }
                """,
    ):
        st.button(
            label=f"{score}",
            key=f"mol_{key}",
            on_click=view_mol_button,
            args=[r_mol],
        )


# Utility functions


def render_error():
    st.markdown(
        "<h2 style='text-align: center;'>Oops. Something went wrong.</h1>",
        unsafe_allow_html=True,
    )
    return None


# Customize Style


def apply_custom_styling():
    custom_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stSlider [data-baseweb=slider]{
            width: 100%;
            margin: 10px;
        }
    .stNumberInput [data-baseweb=input]{
                width: 20%;
            }
    hr {margin: 0px}
    
    </style>
    
    """

    st.html(custom_style)

    return None


def header_image(image_path: str = "./assets/header_background.png"):
    file_ = open(image_path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.html(
        f'<img src="data:image/gif;base64,{data_url}" style="margin: -15px 0 -15px 0; width: 100%; height: 120px; object-fit: cover; object-position: 0 34%;">',
    )
    return None


def stylable_container(key: str, css_styles: str | list[str]) -> "DeltaGenerator":
    """
    Can be used to create buttons with custom styles!


    From streamlit-extras v0.5.5 credit to Lukas Masuch
    Insert a container into your app which you can style using CSS.
    This is useful to style specific elements in your app.

    Args:
        key (str): The key associated with this container. This needs to be unique since all styles will be
            applied to the container with this key.
        css_styles (str | List[str]): The CSS styles to apply to the container elements.
            This can be a single CSS block or a list of CSS blocks.

    Returns:
        DeltaGenerator: A container object. Elements can be added to this container using either the 'with'
            notation or by calling methods directly on the returned object.
    """
    if isinstance(css_styles, str):
        css_styles = [css_styles]

    # Remove unneeded spacing that is added by the style markdown:
    css_styles.append(
        """
> div:first-child {
    margin-bottom: -1rem;
}
"""
    )

    style_text = """
<style>
"""

    for style in css_styles:
        style_text += f"""

div[data-testid="stVerticalBlock"]:has(> div.element-container > div.stMarkdown > div[data-testid="stMarkdownContainer"] > p > span.{key}) {style}

"""

    style_text += f"""
    </style>

<span class="{key}"></span>
"""

    container = st.container()
    container.markdown(style_text, unsafe_allow_html=True)
    return container
