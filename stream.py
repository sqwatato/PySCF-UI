import streamlit as st
import streamlit.components.v1 as components
from pyscf import gto, scf
from streamlit.runtime.scriptrunner import add_script_run_ctx
import threading
import time

if 'queue' not in st.session_state:
    st.session_state['queue'] = []
if 'results' not in st.session_state:
    st.session_state['results'] = []
if 'computing' not in st.session_state:
    st.session_state['computing'] = False


def compute_pyscf(atom, basis_option, verbose_option):
    print(atom)
    print(basis_option)
    print(verbose_option)
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = basis_option
    mol.verbose = int(verbose_option[0])
    mol.output = 'output-test.txt'
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    outputFile = open("output-test.txt", "r")
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

# Function to process the uploaded text file


def process_text_file(uploaded_file):
    if uploaded_file is not None:
        # Read the contents of the file
        text_contents = uploaded_file.getvalue().decode("utf-8")
        return text_contents
    else:
        return None


# Display file uploader for a single text file and processes it
uploaded_file = st.file_uploader("Upload a XYZ input", type=["txt"])
text_contents = process_text_file(uploaded_file)

# Fills xyz_input text area to the contents of the uploaded file
xyz_input = st.text_area(
    "XYZ Input", value=text_contents) if text_contents else st.text_area("XYZ Input")

# Create a Streamlit button which gives example
with st.expander("See Example Input"):
    st.write("C 0.0000000 0.0000000 0.0000000")
    st.write("H 0.6311940 0.6311940 0.6311940")
    st.write("H -0.6311940 -0.6311940 0.6311940")
    st.write("H -0.6311940 0.6311940 -0.6311940")
    st.write("H 0.6311940 -0.6311940 -0.631194")


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
        st.warning(
            "Please provide an XYZ input using the text box or inputting a text file.")

if 'queue' in st.session_state:
    st.subheader("Queue")
    for queue_item in st.session_state['queue']:
        st.write(getMoleculeName(queue_item))

if 'results' in st.session_state:
    st.subheader("Results")
    for result_item in st.session_state['results']:
        st.write(
            f"{result_item[0]} | Energy: {result_item[1]} | Time: {result_item[2]} seconds")

if col3.button('View Log'):
    with open('output-test.txt', 'r') as file:
        log_data = file.read()
        st.markdown(f'```\n{log_data}\n```')

# Computes only if something is added to the queue; grayed out otherwise
compute_disabled = len(st.session_state['queue']) == 0
if col2.button("Compute", disabled=compute_disabled) or st.session_state['computing'] == True:
    if len(st.session_state['queue']) > 0:
        with st.spinner("Computing " + getMoleculeName(st.session_state['queue'][0]) + "..."):
            st.session_state['computing'] = True
            atom = st.session_state['queue'][0]
            st.session_state['queue'].pop(0)
            # st.write("Computing...")
            # progress_text = "Computing..."
            # my_bar = st.progress(0, text=progress_text)

            # for percent_complete in range(100):
            #     time.sleep(0.01)
            #     my_bar.progress(percent_complete + 1, text=progress_text)
            # time.sleep(1)
            # my_bar.empty()
            energy, time_val = compute_pyscf(
                atom, basis_option, verbose_option)
            molecule_name = getMoleculeName(atom)
            st.session_state['results'].append(
                (molecule_name, energy, time_val))
            st.rerun()
    elif st.session_state['computing'] == True:
        st.session_state['computing'] = False
    else:
        st.warning("Please add an XYZ input to the queue.")

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
#                 </html>""")
