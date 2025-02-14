import streamlit as st
from stspeck import Speck

from streamlit.components.v1 import html
import streamlit.components.v1 as components
import ipywidgets as widgets
from ipywidgets import embed

import requests

from streamlit_ketcher import st_ketcher

import base64

from rdkit import Chem
from rdkit.Chem import Draw


# Page setup
# st.set_page_config(
#     page_title="ML Conformer Generator",
#     # page_icon="./frontend/assets/quantori_favicon.ico",
#     layout="wide",
# )
st.title("ML Conformer Generator")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

H2O ='''3
Water molecule
O          0.00000        0.00000        0.11779
H          0.00000        0.75545       -0.47116
H          0.00000       -0.75545       -0.47116'''

# with st.sidebar:
#     ao = st.selectbox("Select ao", [0, 0.1, 0.2, 0.5, 0.8, 1])
#     bonds = st.selectbox("Select bonds", [True, False])
viewer = st.container(height=600)
with viewer:
   res = Speck(H2O)
