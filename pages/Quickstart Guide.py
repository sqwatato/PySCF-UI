import streamlit as st

st.set_page_config(
    page_title="Quickstart Guide",
    page_icon="📝",
)

st.title("Quickstart Guide")
st.write("This UI is intended to serve as an easy and intuitive way to perform self-consistent field calculations, thermodynamic analyses, and harmonic oscillator calculations for molecules using the PySCF library. Here are some instructions on how to use the current features of the UI.")
st.write("PySCF UI was created mainly for future use in organic, biochemical, and pharmaceutical synthesis. As such, all future updates and features will aim to combine general *ab initio* computational method features with aesthetic changes and specialized updates in line with these fields. However, the UI can be used for basic calculations and trend observations for any molecule.")
st.header("Selecting Molecular Geometries")
st.write("The UI allows for both custom .xyz file-type molecular geometries, as well as pre-loaded .xyz geometries. The available pre-loaded geometries can be found under the 'Database' tab in the input section, and have been sourced from the CCCBDB NIST database. The custom geometries can be entered in the 'Text Input' or 'File Input' tabs (also in the input section), and must be in the .xyz format.")
st.write("Future updates will expand the UI to allow for the selection of any .xyz file from the QM9 database, which includes 133885 molecular geometries and is a peer-reviewed database commonly used for ML network training.")
st.write("To search for molecular geometries in the CCCBDB database, please click the link below:")
st.write("https://cccbdb.nist.gov/geom1x.asp")
st.header("Basis Sets")
st.write("PySCF UI currently allows for users to select any of the basis sets built into the PySCF library. These basis sets are Gaussian-type orbital basis sets (PySCF does not have Slater-type orbital basis sets currently built into the library).")
st.write("Please note that not all of the basis sets exist for all the atoms; thus, the UI will throw an error if a basis set has not been found for a particular atom. To see the list of all basis sets offered by PySCF and the atoms for which those basis sets are offered, please visit the link below:")
st.write("https://pyscf.org/_modules/pyscf/gto/basis.html")  
st.write("Future updates will expand the UI to allow for the selection of basis sets from the Basis Stack Exchange (BSE) as well.") 
st.header("Verbose and PySCF Logs")
st.write("The verbose quantity refers to the amount of information displayed in the PySCF logs once the calculation is finished. These logs can be seen under the 'View Logs' tab in the output section. The minimum verbose level provided by the UI is 3; this is the bare minimum that provides the energy and runtime for the molecule. We recommend verbose 5 to get the runtime and energies for each cycle in the SCF calculation process; however, if you would like more data on the process, select a verbose value higher than 5. The maximum verbose value is 9.")
st.write("To see what features are provided at each verbose level, please visit the PySCF developer guide. To get a basic understanding of the features of PySCF (including verbose), please visit the PySCF user guide.")
st.header("Comparative Graphs")
st.write("The comparative graphs, which can be viewed under the 'View Graphs' tab in the output section, allows the user to compare energy and runtime trends for the different molecules they have calculated, with respect to the number of carbons and the total number of atoms. In future updates, this feature may be extended to include trend graphs for thermodynamic and other data as well.")