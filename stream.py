import streamlit as st
from pyscf import gto, scf
from streamlit.runtime.scriptrunner import add_script_run_ctx
import threading
import time

if 'queue' not in st.session_state:
    st.session_state['queue'] = []
if 'results' not in st.session_state:
    st.session_state['results'] = []


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
verbose_option = st.selectbox("Verbose", index=2, options=[
                              "3, energy only", "4, cycles and energy", "5, cycles energy and runtime", "9, max"])

col1, col2, col3 = st.columns(3, gap="small")


def addToQueue(atom):
    st.session_state['queue'].append(atom)


if col1.button("Add to Queue"):
    if xyz_input:
        addToQueue(xyz_input)
    else:
        st.warning("Please provide an XYZ input.")


if col2.button("Compute"):
    if len(st.session_state['queue']) > 0:
        st.write("Computing...")
        for atom in st.session_state['queue']:
            energy, time_val = compute_pyscf(
                atom, basis_option, verbose_option)
            molecule_name = getMoleculeName(atom)
            st.session_state['results'].append(
                (molecule_name, energy, time_val))
            # st.write(
            #     f"Result: {molecule_name} | Energy: {energy} | Time: {time_val} seconds")
    else:
        st.warning("Please add an XYZ input to the queue.")

if st.session_state['results']:
    st.subheader("Results")
    for result_item in st.session_state['results']:
        st.write(
            f"{result_item[0]} | Energy: {result_item[1]} | Time: {result_item[2]} seconds")

if col3.button('View Log'):
    with open('computed/output-test.txt', 'r') as file:
        log_data = file.read()
        st.markdown(f'```\n{log_data}\n```')


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
