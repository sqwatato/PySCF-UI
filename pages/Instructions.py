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
st.write("https://pyscf.org/_modules/pyscf/gto/basis.html")  
st.header("Verbose")
st.write("The verbose quantity refers to the amount of information displayed in the PySCF logs once the calculation is finished. The minimum verbose level provided by the UI is 3; this is the bare minimum that provides the energy and runtime for the molecule. We recommend to use verbose 5 to get the runtime and energies for each cycle in the SCF calculation process; however, if you would like more data on the process, select a verbose value higher than 5. The maximum verbose value is 9.")
