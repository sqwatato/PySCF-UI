import streamlit as st
import streamlit.components.v1 as components
from pyscf import gto, scf
from streamlit.runtime.scriptrunner import add_script_run_ctx
import threading
import time
from stmol import *
import py3Dmol
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdmolfiles import MolFromXYZFile
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import altair as alt
import os
from pyscf.hessian import thermo
from streamlit_extras.row import row
from utils import getAtomicToMoleculeName
# R^2
from sklearn.metrics import r2_score

moleculeNames = getAtomicToMoleculeName()
trend_threshold = 0.95

if 'queue' not in st.session_state:
    st.session_state['queue'] = []
if 'results' not in st.session_state:
    st.session_state['results'] = []
if 'computing' not in st.session_state:
    st.session_state['computing'] = False

# get all files in directory names precomputed_molecules:
precomputed_molecules = list(map(lambda x: x.split(
    ".")[0], os.listdir("precomputed_molecules")))


def compute_pyscf(atom, basis_option, verbose_option, temperature, pressure):
    # print(atom)
    # print(basis_option)
    # print(verbose_option)
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = basis_option
    # mol.verbose = verbose_option
    mol.verbose = int(verbose_option[0])
    mol.output = 'output-test.txt'
    mol.build()

    # mf = scf.RHF(mol)
    # mf.kernel()
    mf =mol.UHF().run()
    hessian = mf.Hessian().kernel()
    harmanalysis = thermo.harmonic_analysis(mf.mol, hessian)
    thermo_info =  thermo.thermo(mf, harmanalysis['freq_au'], temperature, pressure)
    
    outputFile = open("output-test.txt", "r")
    # Extract energy and time information
    time = None
    energy = None
    for line in outputFile.readlines():
        if line.startswith("    CPU time for SCF"):
            time = float(line.split(" ")[-2])

        elif line.startswith("converged SCF energy = "):
            energy = float([i for i in line.split() if i != ''][4])
    data = {
        'energy': energy,
        'time': time,
        'nuclear energy': mf.energy_nuc(),
        'electronic energy': mf.energy_elec(),
        'total energy': mf.energy_tot(),
        # thermodynamics
        'entropy': thermo_info['S_tot'][0],
        'zero point energy': thermo_info['ZPE'][0],
        '0K internal energy': thermo_info['E_0K'][0],
        'internal energy': thermo_info['E_tot'][0],
        'enthalpy': thermo_info['H_tot'][0],
        'gibbs free energy': thermo_info['G_tot'][0],
        # 'Helmholtz free energy': thermo_info['Helmholtz'][0],
        # 'Helmholtz energy': thermo_info['Helmholtz'][0],
    }
    return data


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

# Function to process the uploaded text file


def process_text_file(uploaded_file):
    if uploaded_file is not None:
        # Read the contents of the file
        text_contents = uploaded_file.getvalue().decode("utf-8")
        return text_contents
    else:
        return None


def addToQueue(atom, basis):
    st.session_state['queue'].append((atom, basis))


tabDatabase, tabTextInput, tabFileInput = st.tabs(
    ["Database", "Text Input", "File Input"])

basis_option = st.selectbox(
    "Basis", ["cc-pVTZ", "cc-pVDZ", "cc-pVQZ", "cc-pV5Z"])
verbose_option = st.selectbox("Verbose", index=2, options=[
                              "3, energy only", "4, cycles and energy", "5, cycles energy and runtime", "9, max"])
# verbose_option = st.slider("Verbose", min_value=0, max_value=9, value=2)

#Second Input (NEW) - Pressure of the system
# pressure = 101325 #in Pascals (Pa), 101325 Pa = 1 atm
#Third Input (NEW) - Temperature of the system
# temperature = 298.15 #in K, 298.15K = room temperature (25 degrees Celsius) 
thermo_row = row(2)
temp = thermo_row.number_input("Temperature (K)", min_value=0.0, value=298.15)
press = thermo_row.number_input("Pressure (Pa)", min_value=0.0, value=101325.0)

with tabDatabase:
    selectedMolecule = st.selectbox(
        'Search Molecule Database', precomputed_molecules)
    if st.button('Add to Queue', use_container_width=True, key="db"):
        if selectedMolecule:
            parseDatafile = open(
                "precomputed_molecules/" + selectedMolecule + ".geom.txt", "r").readlines()[4:]
            parseDatafile = "\n".join(parseDatafile[:-1])
            addToQueue(parseDatafile, basis_option)
        else:
            st.warning(
                "Please select a molecule using dropdown menu or inputting a text file.")

with tabTextInput:
    # Create a Streamlit button which gives example
    with st.expander("See Example Input"):
        st.write("C 0.0000000 0.0000000 0.0000000")
        st.write("H 0.6311940 0.6311940 0.6311940")
        st.write("H -0.6311940 -0.6311940 0.6311940")
        st.write("H -0.6311940 0.6311940 -0.6311940")
        st.write("H 0.6311940 -0.6311940 -0.631194")
    # Fills xyz_input text area to the contents of the uploaded file
    xyz_input = st.text_area("XYZ Input", key="textxyz")

    if st.button('Add to Queue', use_container_width=True, key="text"):
        if xyz_input:
            addToQueue(xyz_input, basis_option)
        else:
            st.warning(
                "Please provide an XYZ input using the text box or inputting a text file.")

with tabFileInput:
    # Create a Streamlit button which gives example
    with st.expander("See Example Input"):
        st.write("C 0.0000000 0.0000000 0.0000000")
        st.write("H 0.6311940 0.6311940 0.6311940")
        st.write("H -0.6311940 -0.6311940 0.6311940")
        st.write("H -0.6311940 0.6311940 -0.6311940")
        st.write("H 0.6311940 -0.6311940 -0.631194")
    # Display file uploader for a single text file and processes it
    uploaded_file = st.file_uploader("Upload a XYZ input", type=["txt"])
    text_contents = process_text_file(uploaded_file)
    xyz_input = st.text_area(
        "XYZ Input", value=text_contents, key="filexyz") if text_contents else None
    if st.button('Add to Queue', use_container_width=True, key="filequeue"):
        if text_contents:
            addToQueue(text_contents, basis_option)
        else:
            st.warning(
                "Please provide an XYZ input using file uploader")

col1, col2, col3, col4 = st.columns(4, gap="small")

# if col1.button("Add to Queue"):
#     if xyz_input:
#         addToQueue(xyz_input)
#     else:
#         st.warning(
#             "Please provide an XYZ input using the text box or inputting a text file.")

# Computes only if something is added to the queue; grayed out otherwise
compute_disabled = len(st.session_state['queue']) == 0
if st.button("Compute", disabled=compute_disabled, type="primary", use_container_width=True) or st.session_state['computing'] == True:
    if len(st.session_state['queue']) > 0:
        with st.spinner("Computing " + getMoleculeName(st.session_state['queue'][0][0]) + "..."):
            st.session_state['computing'] = True
            atom = st.session_state['queue'][0][0]
            basis = st.session_state['queue'][0][1]
            st.session_state['queue'].pop(0)
            # st.write("Computing...")
            # progress_text = "Computing..."
            # my_bar = st.progress(0, text=progress_text)

            # for percent_complete in range(100):
            #     time.sleep(0.01)
            #     my_bar.progress(percent_complete + 1, text=progress_text)
            # time.sleep(1)
            # my_bar.empty()

            # Delete empty lines
            parsed = [line for line in atom.splitlines() if line.strip() != ""]
            xyz = "\n".join(parsed)
            mol = f"{len(parsed)}\nname\n{str(xyz)}"

            # output xyz into molecule.xyz file
            with open('molecule.xyz', 'w') as f:
                f.write(f"{len(parsed)}\nhi\n{str(xyz)}")

            raw_mol = MolFromXYZFile('molecule.xyz')
            rdkit_mol = Chem.Mol(raw_mol)
            rdDetermineBonds.DetermineBonds(rdkit_mol, charge=0)

            data = compute_pyscf(
                atom, basis, verbose_option, temp, press)
            data['atoms'] = rdkit_mol.GetNumAtoms()
            data['bonds'] = rdkit_mol.GetNumBonds()
            data['rings'] = rdkit_mol.GetRingInfo().NumRings()
            data['weight'] = Descriptors.MolWt(rdkit_mol)
            data['mol'] = mol
            data['rdkit_mol'] = rdkit_mol
            data['basis'] = basis
            data['molecule_name'] = getMoleculeName(atom)
            
            st.session_state['results'].append(data)
            st.rerun()
            
    elif st.session_state['computing'] == True:
        st.session_state['computing'] = False
    else:
        st.warning("Please add an XYZ input to the queue.")

if 'queue' in st.session_state:
    st.subheader("Queue")
    for queue_item in st.session_state['queue']:
        st.write(f"{getMoleculeName(queue_item[0])} | {queue_item[1]}")


tab1, tab2, tab3 = st.tabs(['Results', 'View Graphs', 'View Logs'])

with tab1:
    if 'results' in st.session_state:
        st.subheader("Results")
        for result_item in st.session_state['results']:
            data = result_item
            with st.container():
                result_col_1, result_col_2 = st.columns([2, 1])
                result_col_1.write(
                    f"{data['molecule_name']} | {data['basis']} | Runtime: {data['time']} seconds")
                result_col_1.write(
                    f"\# of Atoms: {data['atoms']} | \# of Bonds: {data['bonds']} | \# of Rings:  {data['rings']}")
                result_col_1.write(
                    f"Molecular Weight: {data['weight']}")
                # energy data
                result_col_1.write(f"Nuclear Energy: {data['nuclear energy']}")
                result_col_1.write(f"Electronic Energy: {data['electronic energy']}")
                result_col_1.write(f"Total Energy: {data['total energy']}")
                # thermodynamic data
                result_col_1.write(f"Entropy: {data['entropy']}")
                result_col_1.write(f"Zero-Point Energy: {data['zero point energy']}")
                result_col_1.write(f"0K Internal Energy: {data['0K internal energy']}")
                result_col_1.write(f"Internal Energy: {data['internal energy']}")
                result_col_1.write(f"Enthalpy: {data['enthalpy']}")
                result_col_1.write(f"Gibbs Free Energy: {data['gibbs free energy']}")

                with result_col_2:
                    speck_plot(
                        data['mol'], component_h=200, component_w=200, wbox_height="auto", wbox_width="auto")
                # linebreak
                st.write("")
                st.write("")

with tab2:
    # st.subheader("Comparative Graphs (WIP)")
    
    def count_atoms(molecule):
    # Check that there is a valid molecule
        if molecule:

            # Add hydrogen atoms--RDKit excludes them by default
            molecule_with_Hs = Chem.AddHs(molecule)
            comp = defaultdict(lambda: 0)

            # Get atom counts
            for atom in molecule_with_Hs.GetAtoms():
                comp[atom.GetAtomicNum()] += 1

            # # If charged, add charge as "atomic number" 0
            # charge = GetFormalCharge(molecule_with_Hs)
            # if charge != 0:
            #     comp[0] = charge
            return comp

    if 'results' in st.session_state and len(st.session_state['results']) > 1:
        st.subheader("Comparative Graphs")

        independent = [
            'Atoms',
            'Bonds',
            # 'Rings',
            'Weight',
        ]
        dependent = [
            # 'Energy', 
            'Runtime',
            'Nuclear Energy',
            # 'Electronic Energy',
            'Total Energy',
            'Entropy',
            'Zero Point Energy',
            '0K Internal Energy',
            'Internal Energy',
            'Enthalpy',
            'Gibbs Free Energy'
        ]
        
        df = pd.DataFrame({
            'Atoms' : [result_item['atoms'] for result_item in st.session_state['results']],
            'Bonds' : [result_item['bonds'] for result_item in st.session_state['results']],
            # 'Rings' : [result_item['rings'] for result_item in st.session_state['results']],
            'Weight' : [result_item['weight'] for result_item in st.session_state['results']],
            # 'Energy' : [result_item['energy'] for result_item in st.session_state['results']],
            'Runtime' : [result_item['time'] for result_item in st.session_state['results']],
            'Nuclear Energy' : [result_item['nuclear energy'] for result_item in st.session_state['results']],
            # 'Electronic Energy' : [result_item['electronic energy'] for result_item in st.session_state['results']],
            'Total Energy' : [result_item['total energy'] for result_item in st.session_state['results']],
            'Entropy' : [result_item['entropy'] for result_item in st.session_state['results']],
            'Zero Point Energy' : [result_item['zero point energy'] for result_item in st.session_state['results']],
            '0K Internal Energy' : [result_item['0K internal energy'] for result_item in st.session_state['results']],
            'Internal Energy' : [result_item['internal energy'] for result_item in st.session_state['results']],
            'Enthalpy' : [result_item['enthalpy'] for result_item in st.session_state['results']],
            'Gibbs Free Energy' : [result_item['gibbs free energy'] for result_item in st.session_state['results']],
        })
        
        for label in independent:
            for target in dependent:
                print(label, target)
                print(df[label].values, df[target].values)
                # Linear Regression
                coeffs_linear = np.polyfit(
                    df[label].values, df[target].values, 1)
                poly1d_fn_linear = np.poly1d(coeffs_linear)
                x = np.linspace(min(df[label]), max(df[label]), 100)

                # Quadratic Regression
                coeffs_quad = np.polyfit(
                    df[label].values, df[target].values, 2)
                poly1d_fn_quad = np.poly1d(coeffs_quad)
                
                # calculate R^2
                r2_linear = r2_score(df[target], poly1d_fn_linear(df[label]))
                r2_quad = r2_score(df[target], poly1d_fn_quad(df[label]))
                
                if r2_linear >= trend_threshold or r2_quad >= trend_threshold:
                    st.markdown(f'### Number of {label} vs. {target}')
                    # Display Equations
                    st.markdown(
                        f"<span style='color: red;'>Best Fit Linear Equation ({target}): Y = {coeffs_linear[0]:.4f}x + {coeffs_linear[1]:.4f} (R^2 = {r2_linear:.4f})</span>", unsafe_allow_html=True)
                    st.markdown(
                        f"<span style='color: green;'>Best Fit Quadratic Equation ({target}): Y = {coeffs_quad[0]:.4f}x² + {coeffs_quad[1]:.4f}x + {coeffs_quad[2]:.4f} (R^2 = {r2_quad:.4f})</span>", unsafe_allow_html=True)

                    # Create a DataFrame for the regression lines
                    df_line = pd.DataFrame(
                        {label: x, 'Linear': poly1d_fn_linear(x), 'Quadratic': poly1d_fn_quad(x)})

                    # Plot
                    scatter = alt.Chart(df).mark_circle(size=60).encode(
                        x=label,
                        y=target,
                        tooltip=[label, target]
                    )

                    line_linear = alt.Chart(df_line).mark_line(color='red').encode(
                        x=label,
                        y='Linear'
                    )

                    line_quad = alt.Chart(df_line).mark_line(color='green').encode(
                        x=label,
                        y='Quadratic'
                    )

                    # Display the plot
                    st.altair_chart(scatter + line_linear +
                                    line_quad, use_container_width=True)
        
        
        # for atomic_num, count in count_atoms(st.session_state['results'][0]['rdkit_mol']).items():
        
        # atom_counts = [count_atoms(result_item['rdkit_mol'])
        #                for result_item in st.session_state['results']]

        # # Prepare datasets
        # num_atoms = [result_item['atoms']
        #              for result_item in st.session_state['results']]
        # num_bonds = [result_item['bonds'].GetNumBonds()
        #              for result_item in st.session_state['results']]
        # num_conformers = [result_item[4].GetNumConformers()
        #                   for result_item in st.session_state['results']]
        # # 6 and 1 are atomic code
        # num_carbons = [atom_counts[i][6] for i in range(len(atom_counts))]
        # num_hydrogens = [atom_counts[i][1] for i in range(len(atom_counts))]

        # energies = [result_item[1]
        #             for result_item in st.session_state['results']]
        # runtimes = [result_item[2]
        #             for result_item in st.session_state['results']]

        # df_atoms = pd.DataFrame(
        #     {'Atoms': num_atoms, 'Energy': energies, 'Runtime': runtimes})
        # df_bonds = pd.DataFrame(
        #     {'Bonds': num_bonds, 'Energy': energies, 'Runtime': runtimes})
        # df_conformers = pd.DataFrame(
        #     {'Conformers': num_conformers, 'Energy': energies, 'Runtime': runtimes})
        # df_carbons = pd.DataFrame(
        #     {'Carbons': num_carbons, 'Energy': energies, 'Runtime': runtimes})
        # df_hydrogens = pd.DataFrame(
        #     {'Hydrogens': num_hydrogens, 'Energy': energies, 'Runtime': runtimes})

        # Generate Graphs
        # for df, label in zip([df_atoms, df_bonds, df_carbons, df_hydrogens], ['Atoms', 'Bonds', 'Carbons', 'Hydrogens']):
        #     for target in ['Energy', 'Runtime']:
        #         st.markdown(f'### Number of {label} vs. {target}')

        #         # Linear Regression
        #         coeffs_linear = np.polyfit(
        #             df[label].values, df[target].values, 1)
        #         poly1d_fn_linear = np.poly1d(coeffs_linear)
        #         x = np.linspace(min(df[label]), max(df[label]), 100)

        #         # Quadratic Regression
        #         coeffs_quad = np.polyfit(
        #             df[label].values, df[target].values, 2)
        #         poly1d_fn_quad = np.poly1d(coeffs_quad)

        #         # Display Equations
        #         st.markdown(
        #             f"<span style='color: red;'>Best Fit Linear Equation ({target}): Y = {coeffs_linear[0]:.4f}x + {coeffs_linear[1]:.4f}</span>", unsafe_allow_html=True)
        #         st.markdown(
        #             f"<span style='color: green;'>Best Fit Quadratic Equation ({target}): Y = {coeffs_quad[0]:.4f}x² + {coeffs_quad[1]:.4f}x + {coeffs_quad[2]:.4f}</span>", unsafe_allow_html=True)

        #         # Create a DataFrame for the regression lines
        #         df_line = pd.DataFrame(
        #             {label: x, 'Linear': poly1d_fn_linear(x), 'Quadratic': poly1d_fn_quad(x)})

        #         # Plot
        #         scatter = alt.Chart(df).mark_circle(size=60).encode(
        #             x=label,
        #             y=target,
        #             tooltip=[label, target]
        #         )

        #         line_linear = alt.Chart(df_line).mark_line(color='red').encode(
        #             x=label,
        #             y='Linear'
        #         )

        #         line_quad = alt.Chart(df_line).mark_line(color='green').encode(
        #             x=label,
        #             y='Quadratic'
        #         )

        #         # Display the plot
        #         st.altair_chart(scatter + line_linear +
        #                         line_quad, use_container_width=True)

        #     # Display Equation
        #     # st.write(f"Best Fit Equation ({target}): Y = {coeffs[0]:.4f}x + {coeffs[1]:.4f}")

with tab3:
    with open('output-test.txt', 'r') as file:
        log_data = file.read()
        st.markdown(f'```\n{log_data}\n```')


# xyzview = py3Dmol.view(query='pdb:1A2C')
# xyzview.setStyle({'cartoon':{'color':'spectrum'}})
# showmol(xyzview, height = 500,width=800)

# def draw_with_spheres(mol):
#     v = py3Dmol.view(width=300,height=300)
#     IPythonConsole.addMolToView(mol,v)
#     v.zoomTo()
#     v.setStyle({'sphere':{'radius':0.3},'stick':{'radius':0.2}});
#     v.show()


# Attempt at creating an async queue, need to find a way to detect browser closing to stop the queue

# def runQueue():
#     for i in range(1, 10):
#         time.sleep(1)
#         print("test", str(i))


# if 'queue-running' not in st.session_state:
#     st.session_state['queue-running'] = True
#     t = threading.Thread(target=runQueue)
#     add_script_run_ctx(t)
#     t.start()

# components.html("""<html>
# <script>
#     const origClose = window.close;
#     window.close = () => {
#         console.log("asdf");
#         // origClose();
#     }
#     document.addEventListener("beforeunload", () => {
#                 alert(1);
#                 console.log(a.a.a.a);
#     })
# </script>
# <div style="color: white" onclick="">
#                 hihihihi
# </div>
