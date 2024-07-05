import streamlit as st

st.set_page_config(
    page_title="Instructions",
    page_icon="📝",
)

st.title("Instructions")
st.write("This UI is intended to serve as an easy and intuitive way to perform self-consistent field calculations, thermodynamic analyses, and harmonic oscillator calculations for molecules using the PySCF library. Here are some instructions on how to use the current features of the UI.")
st.header("Selecting Molecular Geometries")
st.write("The UI allows for both custom .xyz file-type molecular geometries, as well as pre-loaded .xyz geometries. The available pre-loaded geometries can be found under the 'Database' tab, and have been sourced from the CCCBDB NIST database. The custom geometries can be entered in the 'Text Input' or 'File Input' tabs, and must be in the .xyz format.")
