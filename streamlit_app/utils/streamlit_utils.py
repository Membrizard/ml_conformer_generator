import base64
import os
import re

import matplotlib
import streamlit as st
import streamlit.components.v1 as components
import torch
from mlconfgen import MLConformerGenerator, evaluate_samples
from rdkit import Chem
from rdkit.Chem import Draw

CMAP = matplotlib.cm.get_cmap("viridis")

RESULT_STORAGE = "./tmp"
RESULTS_FILEPATH = f"{RESULT_STORAGE}/generation_results.sdf"

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
def generate_samples_button(
    edm_weights: str, adj_mat_seer_weights: str, ref_mol: Chem.Mol, n_samples: int, n_steps: int, variance: int, device: torch.device
) -> None:
    """
    Generate Samples button callback
    :param edm_weights: a path to the pre-trained EDM block weights
    :param adj_mat_seer_weights: a path to the pre-trained AdjMatSeer block weights
    :param ref_mol: Reference molecule
    :param n_samples: Number of samples to generate
    :param n_steps: Number of diffusion steps for generation
    :param variance: variance in number of heavy atoms for generated molecules
    :param device: Torch device to use for generation
    :return: None
    """
    st.session_state.running = True
    ref, mols = generate_results(
        edm_weights=edm_weights,
        adj_mat_seer_weights=adj_mat_seer_weights,
        ref_mol=ref_mol,
        n_samples=n_samples,
        n_steps=n_steps,
        variance=variance,
        device=device,
    )

    # Save generated molecules in session
    st.session_state.generated_mols = mols

    st.session_state.current_ref = ref

    st.session_state.current_mol = 0

    st.session_state.viewer_update = True

    st.session_state.running = False

    st.rerun()

    return None


def view_mol_button(mol_index: int) -> None:
    """
    View mol button callback
    :param mol_index: index of the molecule from the displayed results
    :return: None
    """
    st.session_state.current_mol = mol_index
    st.session_state.viewer_update = True
    return None


def generate_results(
    edm_weights: str, adj_mat_seer_weights: str, ref_mol: Chem.Mol, n_samples: int, n_steps: int, variance: int, device: torch.device
):
    """
    Generate Samples
    :param edm_weights: a path to the pre-trained EDM block weights
    :param adj_mat_seer_weights: a path to the pre-trained AdjMatSeer block weights
    :param ref_mol: Reference molecule
    :param n_samples: Number of samples to generate
    :param n_steps: Number of diffusion steps for generation
    :param variance: variance in number of heavy atoms for generated molecules
    :param device: Torch device to use for generation
    :return: None
    """
    generator = MLConformerGenerator(
                                     edm_weights=edm_weights,
                                     adj_mat_seer_weights=adj_mat_seer_weights,
                                     diffusion_steps=n_steps,
                                     device=device)
    samples = generator(
        reference_conformer=ref_mol,
        n_samples=n_samples,
        variance=variance,
    )

    aligned_ref, std_samples = evaluate_samples(ref_mol, samples)

    # Sort Samples
    def s_f(x):
        return x["shape_tanimoto"]

    std_samples.sort(key=s_f, reverse=True)

    # Save samples to file
    os.makedirs(RESULT_STORAGE, exist_ok=True)
    writer = Chem.SDWriter(RESULTS_FILEPATH)

    m_ref = Chem.MolFromMolBlock(aligned_ref)
    m_ref.SetProp("_Name", "Reference")

    writer.write(m_ref)
    for i, sample in enumerate(std_samples):
        m_sample = Chem.MolFromMolBlock(sample["mol_block"])
        m_sample.SetProp("_Name", f"MLConfGen_{i + 1}")
        m_sample.SetProp("shape_tanimoto", str(sample["shape_tanimoto"]))
        m_sample.SetProp("chemical_tanimoto", str(sample["chemical_tanimoto"]))
        writer.write(m_sample)

    return aligned_ref, std_samples


def draw_compound_image(compound: Chem.Mol) -> str:
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
) -> None:
    """
    Display search results in grid
    :param mols: molecules to display
    :param c_key: container key
    :param height: container height
    :param cards_per_row: number of molecules per row
    :return: None
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

                components.html(svg_string)
                st.divider()

    return None


def create_view_molecule_button(r_mol: int, score: float, key: int) -> None:
    """
    Create a button for a generated molecule
    :param r_mol: Id of the molecule
    :param score: shape tanimoto score
    :param key: Id of the button
    :return: None

    """
    score = round(score, 2)

    color = CMAP(score)
    hex_color = matplotlib.colors.to_hex(color)

    if score > 0.3:
        l_color = "#262730"
    else:
        l_color = "#d3d3d3"

    with stylable_container(
        key=f"molecule_button_{key}",
        css_styles="""
                button {"""
        + f"\nbackground-color: {hex_color};\n"
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


def header_logo(image_path: str = "./assets/mlconfgen_cosmo_logo"):
    file_ = open(image_path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.html(
        f'<img src="data:image/gif;base64,{data_url}" style="margin: -100px 0px -60px -60px; height: 300px; object-fit: cover; object-position: 0 0;">',
    )
    return None


def stylable_container(key: str, css_styles: str | list[str]):
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
