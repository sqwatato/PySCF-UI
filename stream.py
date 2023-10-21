import streamlit as st
from pyscf import gto, scf
import time

queue = []
results = []

def compute_pyscf(atom, basis_option, verbose_option):
    print(atom)
    print(basis_option)
    print(verbose_option)
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = basis_option
    mol.verbose = int(verbose_option[0])
    mol.output = 'computed/output-test.txt'
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    outputFile = open("computed/output-test.txt", "r")
    # Extract energy and time information
    time = None
    energy = None
    for line in outputFile.readlines():
        if line.startswith("    CPU time for SCF"):
            time = float(line.split(" ")[-2])

        elif line.startswith("converged SCF energy = "):
            energy = float(line.split(" ")[-1])
    return energy, time

def getMoleculeName(atom):
    d = {}
    for line in atom.split("\n"):
        try:
            name = line.split()[0]
            if name in d:
                d[name] += 1
            else:
                d[name] = 1
        except:
            pass
    name = ""
    for key in d:
        name += key + str(d[key])
    return name

# Streamlit layout
st.title("PySCF")

xyz_input = st.text_area("XYZ Input")
basis_option = st.selectbox("Basis", ["cc-pVTZ", "asdf"])
verbose_option = st.selectbox("Verbose", ["3, energy only", "4, cycles and energy", "5, cycles energy and runtime", "9, max"])

# Create two columns for the buttons
col1, col2 = st.columns(2, gap="small")

if col1.button("Compute"):
    if xyz_input:
        st.write("Computing...")
        energy, time_val = compute_pyscf(xyz_input, basis_option, verbose_option)
        
        # Add the results to your queue and update the UI
        molecule_name = getMoleculeName(xyz_input)
        results.append((molecule_name, energy, time_val))
        st.write(f"Result: {molecule_name} | Energy: {energy} | Time: {time_val} seconds")
    else:
        st.warning("Please provide an XYZ input.")

if results:
    st.subheader("Results")
    for result_item in results:
        st.write(f"{result_item[0]} | Energy: {result_item[1]} | Time: {result_item[2]} seconds")

if col2.button('View Log'):
    with open('computed/output-test.txt', 'r') as file:
        log_data = file.read()
        st.markdown(f'```\n{log_data}\n```')
