import streamlit as st

st.set_page_config(
    page_title="Instructions",
    page_icon="📝",
)

st.title("Instructions")
st.write("This UI is intended to serve as an easy and intuitive way to perform self-consistent field calculations, thermodynamic analyses, and harmonic oscillator calculations for molecules using the PySCF library. Here are some instructions on how to use the current features of the UI.")
st.header("Selecting Molecular Geometries")
st.write("The UI allows for both custom .xyz file-type molecular geometries, as well as pre-loaded .xyz geometries. The available pre-loaded geometries can be found under the 'Database' tab, and have been sourced from the CCCBDB NIST database. The custom geometries can be entered in the 'Text Input' or 'File Input' tabs, and must be in the .xyz format.")
st.header("Basis Sets")
st.write("PySCF UI currently allows for users to select any of the basis sets built into the PySCF library. These basis sets are Gaussian-type orbital basis sets (PySCF does not have Slater-type orbital basis sets currently built into the library).")
st.write("Please note that not all of the basis sets exist for all the atoms; thus, the UI will throw an error if a basis set has not been found for a particular atom. To see the list of all basis sets offered by PySCF, and the atoms for which those basis sets are offered, please visit the link below:")
         
