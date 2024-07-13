import streamlit as st
import streamlit.components.v1 as components
from pyscf import gto, scf, dft
from streamlit.runtime.scriptrunner import add_script_run_ctx
import threading
import time
from stmol import *
import py3Dmol
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdmolfiles import MolFromXYZFile
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Draw import MolToImage
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
import requests
import timeit
import basis_set_exchange as bse

st.set_page_config(
    page_title="PySCF UI",
    page_icon="📈",
)

# api_url = "http://0.0.0.0:8000/calculate"
moleculeNames = getAtomicToMoleculeName()
trend_threshold = 0.95

if 'queue' not in st.session_state:
    st.session_state['queue'] = []
if 'results' not in st.session_state:
    st.session_state['results'] = []
if 'computing' not in st.session_state:
    st.session_state['computing'] = False
if 'counter' not in st.session_state:
    st.session_state['counter'] = 0

# get all files in directory names precomputed_molecules:
precomputed_molecules = list(map(lambda x: x.split(
    ".")[0], os.listdir("precomputed_molecules")))


def compute_pyscf(atom, basis_source, basis_option, verbose_option, method, temperature, pressure):
    # print(atom)
    # print(basis_option)
    # print(verbose_option)
    mol = gto.Mole()
    mol.atom = atom
    if basis_source == "PySCF":
        mol.basis = basis_option
    elif basis_source == "BSE":
        mol.basis = {'H':gto.basis.parse(bse.get_basis(basis_option,elements=[1],fmt='nwchem',header=False)), 
             'C':gto.basis.parse(bse.get_basis(basis_option,elements=[6],fmt='nwchem',header=False)),
             'N':gto.basis.parse(bse.get_basis(basis_option,elements=[7],fmt='nwchem',header=False)),
             'O':gto.basis.parse(bse.get_basis(basis_option,elements=[8],fmt='nwchem',header=False)),
             'F':gto.basis.parse(bse.get_basis(basis_option,elements=[9],fmt='nwchem',header=False)),
             'P':gto.basis.parse(bse.get_basis(basis_option,elements=[15],fmt='nwchem',header=False)),
             'S':gto.basis.parse(bse.get_basis(basis_option,elements=[16],fmt='nwchem',header=False))}
    mol.verbose = verbose_option
    # mol.verbose = int(verbose_option[0])
    mol.output = 'output-test.txt'
    mol.build()

    # mf = scf.RHF(mol)
    # mf.kernel()
    if method == "UHF":
        mf = scf.UHF(mol)
        mf.kernel()
    elif method == "UKS":
        mf = dft.UKS(mol)
        mf.kernel()
    hessian = mf.Hessian().kernel()
    harmanalysis = thermo.harmonic_analysis(mf.mol, hessian)
    thermo_info =  thermo.thermo(mf, harmanalysis['freq_au'], temperature, pressure)
    
    outputFile = open("output-test.txt", "r")
    # Extract energy and time information
    scf_cpu_time = None
    scf_wall_time = None
    hessian_cpu_time = None
    hessian_wall_time = None
    energy = None
    for line in outputFile.readlines():
        if line.startswith("    CPU time for SCF"):
            words = [i for i in line.split() if i]
            # ['CPU', 'time', 'for', 'SCF', '3.00', 'sec,', 'wall', 'time', '0.51', 'sec']
            scf_cpu_time = float(words[4])
            scf_wall_time = float(words[8])

        # elif line.startswith("converged SCF energy = "):
        #     energy = float([i for i in line.split() if i != ''][4])
        elif line.startswith("    CPU time for UHF hessian"):
            words = [i for i in line.split() if i]
            # ['CPU', 'time', 'for', 'UHF', 'hessian', '7.12', 'sec,', 'wall', 'time', '4.87', 'sec']
            hessian_cpu_time = float(words[5])
            hessian_wall_time = float(words[9])
    
    #Helmholtz Free Energy
    F_elec = (thermo_info['E_elec'][0] - temperature * thermo_info['S_elec' ][0], 'Eh')
    F_trans = (thermo_info['E_trans'][0] - temperature * thermo_info['S_trans'][0], 'Eh')
    F_rot = (thermo_info['E_rot'][0] - temperature * thermo_info['S_rot'][0], 'Eh')
    F_vib = (thermo_info['E_vib'][0] - temperature * thermo_info['S_vib'][0], 'Eh')
    F_tot = (F_elec[0] + F_trans[0] + F_rot[0] + F_vib[0], 'Eh') 
    
    #Massieu Potential/Helmholtz Free Entropy
    Φ_elec = (F_elec[0]/(-1*temperature), 'Eh/K')
    Φ_trans = (F_trans[0]/(-1*temperature), 'Eh/K')
    Φ_rot = (F_rot[0]/(-1*temperature), 'Eh/K')
    Φ_vib = (F_vib[0]/(-1*temperature), 'Eh/K')
    Φ_tot = (F_tot[0]/(-1*temperature), 'Eh/K')    
    
    #Planck Potential/Gibbs Free Entropy
    Ξ_elec = (thermo_info['G_elec'][0]/(-1*temperature), 'Eh/K')
    Ξ_trans = (thermo_info['G_trans'][0]/(-1*temperature), 'Eh/K')
    Ξ_rot = (thermo_info['G_rot'][0]/(-1*temperature), 'Eh/K')
    Ξ_vib = (thermo_info['G_vib'][0]/(-1*temperature), 'Eh/K')
    Ξ_tot = (thermo_info['G_tot'][0]/(-1*temperature), 'Eh/K')   
    
    data = {
        # 'energy': energy,
        'SCF CPU Runtime': scf_cpu_time,
        'SCF Wall Runtime': scf_wall_time,
        'Hessian CPU Runtime': hessian_cpu_time,
        'Hessian Wall Runtime': hessian_wall_time,
        'Converged SCF Nuclear Energy (Ha)': mf.energy_nuc(),
        'Converged SCF Electronic Energy (Ha)': mf.energy_elec(),
        'Converged SCF Total Energy (Ha)': mf.energy_tot(),
        # thermodynamic 
        # Heat Capacity
        'Constant Volume Heat Capacity (Ha/K)': thermo_info['Cv_tot'][0],
        'Constant Pressure Heat Capacity (Ha/K)': thermo_info['Cp_tot'][0],
        'Zero-Point Energy (Ha)': thermo_info['ZPE'][0],
        '0K Internal Energy (Ha)': thermo_info['E_0K'][0],
        'Internal Energy (at given T) (Ha)': thermo_info['E_tot'][0],
            'Electronic Internal Energy (Ha)': thermo_info['E_elec'][0],
            'Vibrational Internal Energy (Ha)': thermo_info['E_vib'][0],
            'Translational Internal Energy (Ha)': thermo_info['E_trans'][0],
            'Rotational Internal Energy (Ha)': thermo_info['E_rot'][0],
        # enthalpy
        'Enthalpy (Ha)': thermo_info['H_tot'][0],
            'Electronic Enthalpy (Ha)': thermo_info['H_elec'][0],
            'Vibrational Enthalpy (Ha)': thermo_info['H_vib'][0],
            'Translational Enthalpy (Ha)': thermo_info['H_trans'][0],
            'Rotational Enthalpy (Ha)': thermo_info['H_rot'][0],
        # gibbs free energy
        'Gibbs Free Energy (Ha)': thermo_info['G_tot'][0],
            'Electronic Gibbs Free Energy (Ha)': thermo_info['G_elec'][0],
            'Vibrational Gibbs Free Energy (Ha)': thermo_info['G_vib'][0],
            'Translational Gibbs Free Energy (Ha)': thermo_info['G_trans'][0],
            'Rotational Gibbs Free Energy (Ha)': thermo_info['G_rot'][0],
        # Helmholtz free energy
        'Helmholtz Free Energy (Ha)': F_tot[0],
            'Electronic Helmholtz Free Energy (Ha)': F_elec[0],
            'Vibrational Helmholtz Free Energy (Ha)': F_vib[0],
            'Translational Helmholtz Free Energy (Ha)': F_trans[0],
            'Rotational Helmholtz Free Energy (Ha)': F_rot[0],
        # Entropy
        'Entropy (Ha/K)': thermo_info['S_tot'][0],
            'Electronic Entropy (Ha/K)': thermo_info['S_elec'][0],
            'Vibrational Entropy (Ha/K)': thermo_info['S_vib'][0],
            'Translational Entropy (Ha/K)': thermo_info['S_trans'][0],
            'Rotational Entropy (Ha/K)': thermo_info['S_rot'][0],
        # Massieu Potential/Helmholtz Free Entropy
        'Massieu Potential/Helmholtz Free Potential (Ha/K)': Φ_tot[0],
            'Electronic Massieu Potential/Helmholtz Free Potential (Ha/K)': Φ_elec[0],
            'Vibrational Massieu Potential/Helmholtz Free Potential (Ha/K)': Φ_vib[0],
            'Translational Massieu Potential/Helmholtz Free Potential (Ha/K)': Φ_trans[0],
            'Rotational Massieu Potential/Helmholtz Free Potential (Ha/K)': Φ_rot[0],
        # Planck Potential/Gibbs Free Entropy
        'Planck Potential/Gibbs Free Potential (Ha/K)': Ξ_tot[0],
            'Electronic Planck Potential/Gibbs Free Potential (Ha/K)': Ξ_elec[0],
            'Vibrational Planck Potential/Gibbs Free Potential (Ha/K)': Ξ_vib[0],
            'Translational Planck Potential/Gibbs Free Potential (Ha/K)': Ξ_trans[0],
            'Rotational Planck Potential/Gibbs Free Potential (Ha/K)': Ξ_rot[0],
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
    for key,value in d.items():
        if value > 1:
            name += key + str(value)
        else:
            name += key
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


tabQM9Database, tabCCCBDBDatabase, tabTextInput, tabFileInput = st.tabs(
    ["QM9 Database", "CCCBDB PySCF UI Database", "Text Input", "File Input"])
method_option = st.selectbox(
    "Method", ["UKS","UHF"])
bse_pyscf = st.radio("Source of Basis Sets",['PySCF','BSE'])
if bse_pyscf == 'PySCF':
    basis_option = st.selectbox(
    "Basis", ['ano','anorcc', 'anoroosdz', 'anoroostz', 'roosdz', 'roostz', 'ccpvdz', 'ccpvtz', 'ccpvqz', 'ccpv5z', 'ccpvdpdz', 'augccpvdz', 'augccpvtz', 'augccpvqz', 'augccpv5z', 'augccpvdpdz', 'ccpvdzdk', 'ccpvtzdk', 'ccpvqzdk', 'ccpv5zdk', 'ccpvdzdkh', 'ccpvtzdkh', 'ccpvqzdkh', 'ccpv5zdkh', 'augccpvdzdk', 'augccpvtzdk', 'augccpvqzdk', 'augccpv5zdk', 'augccpvdzdkh', 'augccpvtzdkh', 'augccpvqzdkh', 'augccpv5zdkh', 'ccpvdzjkfit', 'ccpvtzjkfit', 'ccpvqzjkfit', 'ccpv5zjkfit', 'ccpvdzri', 'ccpvtzri', 'ccpvqzri', 'ccpv5zri', 'augccpvdzjkfit', 'augccpvdzpjkfit', 'augccpvtzjkfit', 'augccpvqzjkfit', 'augccpv5zjkfit', 'heavyaugccpvdzjkfit', 'heavyaugccpvtzjkfit', 'heavyaugccpvdzri', 'heavyaugccpvtzri', 'augccpvdzri', 'augccpvdzpri', 'augccpvqzri', 'augccpvtzri', 'augccpv5zri', 'ccpvtzdk3', 'ccpvqzdk3', 'augccpvtzdk3', 'augccpvqzdk3', 'dyalldz', 'dyallqz', 'dyalltz', 'faegredz', 'iglo', 'iglo3', '321++g', '321++g*', '321++gs', '321g', '321g*', '321gs', '431g', '631++g', '631++g*', '631++gs', '631++g**', '631++gss', '631+g', '631+g*', '631+gs', '631+g**', '631+gss', '6311++g', '6311++g*', '6311++gs', '6311++g**', '6311++gss', '6311+g', '6311+g*', '6311+gs', '6311+g**', '6311+gss', '6311g', '6311g*', '6311gs', '6311g**', '6311gss', '631g', '631g*', '631gs', '631g**', '631gss', 'sto3g', 'sto6g', 'minao', 'dz', 'dzpdunning', 'dzvp', 'dzvp2', 'dzp', 'tzp', 'qzp', 'adzp', 'atzp', 'aqzp', 'dzpdk', 'tzpdk', 'qzpdk', 'dzpdkh', 'tzpdkh', 'qzpdkh', 'def2svp', 'def2svpd', 'def2tzvpd', 'def2tzvppd', 'def2tzvpp', 'def2tzvp', 'def2qzvpd', 'def2qzvppd', 'def2qzvpp', 'def2qzvp', 'def2svpjfit', 'def2svpjkfit', 'def2tzvpjfit', 'def2tzvpjkfit', 'def2tzvppjfit', 'def2tzvppjkfit', 'def2qzvpjfit', 'def2qzvpjkfit', 'def2qzvppjfit', 'def2qzvppjkfit', 'def2universaljfit', 'def2universaljkfit', 'def2svpri', 'def2svpdri' , 'def2tzvpri', 'def2tzvpdri', 'def2tzvppri', 'def2tzvppdri', 'def2qzvpri' , 'def2qzvppri', 'def2qzvppdri', 'tzv', 'weigend', 'weigend+etb', 'weigendcfit', 'weigendjfit', 'weigendjkfit', 'demon', 'demoncfit', 'ahlrichs', 'ahlrichscfit', 'ccpvtzfit', 'ccpvdzfit', 'ccpwcvtzmp2fit', 'ccpvqzmp2fit', 'ccpv5zmp2fit', 'augccpwcvtzmp2fit', 'augccpvqzmp2fit', 'augccpv5zmp2fit', 'ccpcvdz' , 'ccpcvtz', 'ccpcvqz', 'ccpcv5z', 'ccpcv6z', 'ccpwcvdz', 'ccpwcvtz', 'ccpwcvqz', 'ccpwcv5z', 'ccpwcvdzdk', 'ccpwcvtzdk', 'ccpwcvqzdk', 'ccpwcv5zdk', 'ccpwcvtzdk3', 'ccpwcvqzdk3', 'augccpwcvdz', 'augccpwcvtz', 'augccpwcvqz', 'augccpwcv5z', 'augccpwcvtzdk', 'augccpwcvqzdk', 'augccpwcv5zdk', 'augccpwcvtzdk3', 'augccpwcvqzdk3', 'dgaussa1cfit', 'dgaussa1xfit', 'dgaussa2cfit', 'dgaussa2xfit', 'ccpvdzpp', 'ccpvtzpp', 'ccpvqzpp', 'ccpv5zpp', 'crenbl', 'crenbs', 'lanl2dz', 'lanl2tz', 'lanl08','sbkjc','stuttgart', 'stuttgartdz', 'stuttgartrlc', 'stuttgartrsc', 'stuttgartrsc_mdf', 'ccpwcvdzpp', 'ccpwcvtzpp', 'ccpwcvqzpp', 'ccpwcv5zpp', 'ccpvdzppnr', 'ccpvtzppnr', 'augccpvdzpp', 'augccpvtzpp', 'augccpvqzpp', 'augccpv5zpp', 'pc0', 'pc1', 'pc2', 'pc3', 'pc4' 'augpc0', 'augpc1', 'augpc2', 'augpc3', 'augpc4', 'pcseg0', 'pcseg1', 'pcseg2', 'pcseg3', 'pcseg4', 'augpcseg0', 'augpcseg1', 'augpcseg2', 'augpcseg3', 'augpcseg4', 'sarcdkh', 'bfdvdz', 'bfdvtz', 'bfdvqz', 'bfdv5z', 'bfd', 'bfdpp', 'ccpcvdzf12optri', 'ccpcvtzf12optri', 'ccpcvqzf12optri', 'ccpvdzf12optri', 'ccpvtzf12optri', 'ccpvqzf12optri', 'ccpv5zf12', 'ccpvdzf12rev2', 'ccpvtzf12rev2', 'ccpvqzf12rev2', 'ccpv5zf12rev2', 'ccpvdzf12nz', 'ccpvtzf12nz', 'ccpvqzf12nz', 'augccpvdzoptri', 'augccpvtzoptri', 'augccpvqzoptri', 'augccpv5zoptri', 'pobtzvp', 'pobtzvpp', 'crystalccpvdz', 'ccecp', 'ccecpccpvdz', 'ccecpccpvtz', 'ccecpccpvqz', 'ccecpccpv5z', 'ccecpccpv6z', 'ccecpaugccpvdz', 'ccecpaugccpvtz', 'ccecpaugccpvqz', 'ccecpaugccpv5z', 'ccecpaugccpv6z', 'ccecphe', 'ccecpheccpvdz', 'ccecpheccpvtz', 'ccecpheccpvqz', 'ccecpheccpv5z', 'ccecpheccpv6z', 'ccecpheaugccpvdz', 'ccecpheaugccpvtz', 'ccecpheaugccpvqz', 'ccecpheaugccpv5z', 'ccecpheaugccpv6z', 'ccecpreg', 'ccecpregccpvdz', 'ccecpregccpvtz', 'ccecpregccpvqz', 'ccecpregccpv5z', 'ccecpregaugccpvdz', 'ccecpregaugccpvtz', 'ccecpregaugccpvqz', 'ccecpregaugccpv5z', 'ecpds10mdfso', 'ecpds28mdfso', 'ecpds28mwbso', 'ecpds46mdfso', 'ecpds60mdfso', 'ecpds60mwbso', 'ecpds78mdfso', 'ecpds92mdfbso', 'ecpds92mdfbqso','gthaugdzvp', 'gthaugqzv2p', 'gthaugqzv3p', 'gthaugtzv2p', 'gthaugtzvp', 'gthdzv', 'gthdzvp', 'gthqzv2p', 'gthqzv3p', 'gthszv', 'gthtzv2p', 'gthtzvp', 'gthccdzvp', 'gthcctzvp', 'gthccqzvp', 'gthszvmolopt', 'gthdzvpmolopt', 'gthtzvpmolopt', 'gthtzv2pmolopt', 'gthszvmoloptsr', 'gthdzvpmoloptsr', 'gthblyp', 'gthbp', 'gthhcth120' 'gthhcth407', 'gtholyp', 'gthlda', 'gthpade', 'gthpbe', 'gthpbesol', 'gthhf', 'gthhfrev'], index=101)
# verbose_option = st.selectbox("Verbose", index=2, options=[
                            #   "3, energy only", "4, cycles and energy", "5, cycles energy and runtime", "6", "7", "8", "9, max"])
    st.write("*Please note that the PySCF basis sets may not have sets for the particular atoms in the queued molecule. If the selected basis set cannot be found for any atom in the molecule, the UI will return an error. For more information on this, please see the Quickstart Guide.*")
elif bse_pyscf == 'BSE':
    basis_option = st.selectbox(
    "Basis", ['2ZaPa-NR', '2ZaPa-NR-CV', '3-21G', '3ZaPa-NR', '3ZaPa-NR-CV', '4-31G', '4ZaPa-NR', '4ZaPa-NR-CV', '5ZaPa-NR', '5ZaPa-NR-CV', '6-21G', '6-31++G', '6-31++G*', '6-31++G**', '6-31+G', '6-31+G*', '6-31+G**', '6-311++G', '6-311++G(2d,2p)', '6-311++G(3df,3pd)', '6-311++G*', '6-311++G**', '6-311+G', '6-311+G(2d,p)', '6-311+G*', '6-311+G**', '6-311G', '6-311G(d,p)', '6-311G*', '6-311G**', '6-311G**-RIFIT', '6-311xxG(d,p)', '6-31G', '6-31G(2df,p)', '6-31G(3df,3pd)', '6-31G(d,p)', '6-31G*', '6-31G**', '6-31G**-RIFIT', '6ZaPa-NR', '7ZaPa-NR', 'admm-1', 'admm-2', 'admm-3', 'AHGBS-5', 'AHGBS-7', 'AHGBS-9', 'AHGBSP1-5', 'AHGBSP1-7', 'AHGBSP1-9', 'AHGBSP2-5', 'AHGBSP2-7', 'AHGBSP2-9', 'AHGBSP3-5', 'AHGBSP3-7', 'AHGBSP3-9', 'Ahlrichs pVDZ', 'Ahlrichs VDZ', 'Ahlrichs VTZ', 'ANO-DK3', 'ano-pV5Z', 'ano-pVDZ', 'ano-pVQZ', 'ano-pVTZ', 'ANO-R', 'ANO-R0', 'ANO-R1', 'ANO-R2', 'ANO-R3', 'ANO-RCC', 'ANO-RCC-MB', 'ANO-RCC-VDZ', 'ANO-RCC-VDZP', 'ANO-RCC-VQZP', 'ANO-RCC-VTZP', 'ANO-VT-DZ', 'ANO-VT-QZ', 'ANO-VT-TZ', 'apr-cc-pV(Q+d)Z', 'ATZP-ZORA', 'aug-admm-1', 'aug-admm-2', 'aug-admm-3', 'aug-ano-pV5Z', 'aug-ano-pVDZ', 'aug-ano-pVQZ', 'aug-ano-pVTZ', 'aug-cc-pV(5+d)Z', 'aug-cc-pV(D+d)Z', 'aug-cc-pV(Q+d)Z', 'aug-cc-pV(T+d)Z', 'aug-cc-pV5Z', 'aug-cc-pV5Z-DK', 'aug-cc-pV5Z-OPTRI', 'aug-cc-pV5Z-RIFIT', 'aug-cc-pV6Z', 'aug-cc-pV6Z-RIFIT', 'aug-cc-pVDZ', 'aug-cc-pVDZ-DK', 'aug-cc-pVDZ-OPTRI', 'aug-cc-pVDZ-RIFIT', 'aug-cc-pVQZ', 'aug-cc-pVQZ-DK', 'aug-cc-pVQZ-OPTRI', 'aug-cc-pVQZ-RIFIT', 'aug-cc-pVTZ', 'aug-cc-pVTZ-DK', 'aug-cc-pVTZ-J', 'aug-cc-pVTZ-OPTRI', 'aug-cc-pVTZ-RIFIT', 'aug-pc-0', 'aug-pc-1', 'aug-pc-2', 'aug-pc-3', 'aug-pc-4', 'aug-pcH-1', 'aug-pcH-2', 'aug-pcH-3', 'aug-pcH-4', 'aug-pcJ-0', 'aug-pcJ-0_2006', 'aug-pcJ-1', 'aug-pcJ-1_2006', 'aug-pcJ-2', 'aug-pcJ-2_2006', 'aug-pcJ-3', 'aug-pcJ-3_2006', 'aug-pcJ-4', 'aug-pcJ-4_2006', 'aug-pcS-0', 'aug-pcS-1', 'aug-pcS-2', 'aug-pcS-3', 'aug-pcS-4', 'aug-pcseg-0', 'aug-pcseg-1', 'aug-pcseg-2', 'aug-pcseg-3', 'aug-pcseg-4', 'aug-pcSseg-0', 'aug-pcSseg-1', 'aug-pcSseg-2', 'aug-pcSseg-3', 'aug-pcSseg-4', 'CADPAC-TZ2P', 'cc-pV(5+d)Z', 'cc-pV(D+d)Z', 'cc-pV(Q+d)Z', 'cc-pV(T+d)Z', 'cc-pV5Z', 'cc-pV5Z(fi/sf/fw)', 'cc-pV5Z(fi/sf/lc)', 'cc-pV5Z(fi/sf/sc)', 'cc-pV5Z(pt/sf/fw)', 'cc-pV5Z(pt/sf/lc)', 'cc-pV5Z(pt/sf/sc)', 'cc-pV5Z-DK', 'cc-pV5Z-F12', 'cc-pV5Z-JKFIT', 'cc-pV5Z-RIFIT', 'cc-pV6Z', 'cc-pV6Z-RIFIT', 'cc-pVDZ', 'cc-pVDZ(fi/sf/fw)', 'cc-pVDZ(fi/sf/lc)', 'cc-pVDZ(fi/sf/sc)', 'cc-pVDZ(pt/sf/fw)', 'cc-pVDZ(pt/sf/lc)', 'cc-pVDZ(pt/sf/sc)', 'cc-pVDZ(seg-opt)', 'cc-pVDZ-DK', 'cc-pVDZ-F12', 'cc-pVDZ-F12-OPTRI', 'cc-pVDZ-F12-OPTRI+', 'cc-pVDZ-RIFIT', 'cc-pVQZ', 'cc-pVQZ(fi/sf/fw)', 'cc-pVQZ(fi/sf/lc)', 'cc-pVQZ(fi/sf/sc)', 'cc-pVQZ(pt/sf/fw)', 'cc-pVQZ(pt/sf/lc)', 'cc-pVQZ(pt/sf/sc)', 'cc-pVQZ-DK', 'cc-pVQZ-F12', 'cc-pVQZ-F12-OPTRI', 'cc-pVQZ-F12-OPTRI+', 'cc-pVQZ-JKFIT', 'cc-pVQZ-RIFIT', 'cc-pVTZ', 'cc-pVTZ(fi/sf/fw)', 'cc-pVTZ(fi/sf/lc)', 'cc-pVTZ(fi/sf/sc)', 'cc-pVTZ(pt/sf/fw)', 'cc-pVTZ(pt/sf/lc)', 'cc-pVTZ(pt/sf/sc)', 'cc-pVTZ(seg-opt)', 'cc-pVTZ-DK', 'cc-pVTZ-F12', 'cc-pVTZ-F12-OPTRI', 'cc-pVTZ-F12-OPTRI+', 'cc-pVTZ-JKFIT', 'cc-pVTZ-RIFIT', 'ccemd-2', 'ccemd-3', 'coemd-2', 'coemd-3', 'coemd-4', 'coemd-ref', 'CRENBL', 'def2-mTZVP', 'def2-mTZVPP', 'def2-mTZVPP-RIJ', 'def2-QZVP', 'def2-QZVP-RIFIT', 'def2-QZVPD', 'def2-QZVPP', 'def2-QZVPP-RIFIT', 'def2-QZVPPD', 'def2-QZVPPD-RIFIT', 'def2-SV(P)', 'def2-SV(P)-JKFIT', 'def2-SV(P)-RIFIT', 'def2-SVP', 'def2-SVP-RIFIT', 'def2-SVPD', 'def2-SVPD-RIFIT', 'def2-TZVP', 'def2-TZVP-RIFIT', 'def2-TZVPD', 'def2-TZVPD-RIFIT', 'def2-TZVPP', 'def2-TZVPP-RIFIT', 'def2-TZVPPD', 'def2-TZVPPD-RIFIT', 'def2-universal-JFIT', 'def2-universal-JKFIT', 'deMon2k-DZVP-GGA', 'DFO+-NRLMOL', 'DFO-NRLMOL', 'dgauss-a1-dftjfit', 'dgauss-a1-dftxfit', 'dgauss-a2-dftjfit', 'dgauss-a2-dftxfit', 'dgauss-dzvp', 'dgauss-dzvp2', 'dgauss-tzvp', 'DZ (Dunning-Hay)', 'DZP (Dunning-Hay)', 'HGBS-5', 'HGBS-7', 'HGBS-9', 'HGBSP1-5', 'HGBSP1-7', 'HGBSP1-9', 'HGBSP2-5', 'HGBSP2-7', 'HGBSP2-9', 'HGBSP3-5', 'HGBSP3-7', 'HGBSP3-9', 'IGLO-II', 'IGLO-III', 'jgauss-dzp', 'jgauss-qz2p', 'jgauss-tzp1', 'jgauss-tzp2', 'jorge-5ZP', 'jorge-5ZP-DKH', 'jorge-6ZP', 'jorge-6ZP-DKH', 'jorge-A5ZP', 'jorge-ADZP', 'jorge-AQZP', 'jorge-ATZP', 'jorge-DZP', 'jorge-DZP-DKH', 'jorge-QZP', 'jorge-QZP-DKH', 'jorge-TZP', 'jorge-TZP-DKH', 'jul-cc-pV(D+d)Z', 'jul-cc-pV(Q+d)Z', 'jul-cc-pV(T+d)Z', 'jun-cc-pV(D+d)Z', 'jun-cc-pV(Q+d)Z', 'jun-cc-pV(T+d)Z', 'Koga unpolarized', 'LANL2DZ', 'LANL2DZdp', 'maug-cc-pV(D+d)Z', 'maug-cc-pV(Q+d)Z', 'maug-cc-pV(T+d)Z', 'may-cc-pV(Q+d)Z', 'may-cc-pV(T+d)Z', 'MIDI', 'MIDI!', 'MIDIX', 'MINI', 'NLO-V', 'NMR-DKH (TZ2P)', 'pc-0', 'pc-1', 'pc-2', 'pc-3', 'pc-4', 'pcemd-2', 'pcemd-3', 'pcemd-4', 'pcH-1', 'pcH-2', 'pcH-3', 'pcH-4', 'pcJ-0', 'pcJ-0_2006', 'pcJ-1', 'pcJ-1_2006', 'pcJ-2', 'pcJ-2_2006', 'pcJ-3', 'pcJ-3_2006', 'pcJ-4', 'pcJ-4_2006', 'pcS-0', 'pcS-1', 'pcS-2', 'pcS-3', 'pcS-4', 'pcseg-0', 'pcseg-1', 'pcseg-2', 'pcseg-3', 'pcseg-4', 'pcSseg-0', 'pcSseg-1', 'pcSseg-2', 'pcSseg-3', 'pcSseg-4', 'pob-DZVP-rev2', 'pob-TZVP', 'pob-TZVP-rev2', 'Roos Augmented Double Zeta ANO', 'Roos Augmented Triple Zeta ANO', 'Sadlej pVTZ', 'sap_grasp_large', 'sap_grasp_small', 'sap_helfem_large', 'sap_helfem_small', 'Sapporo-DZP', 'Sapporo-DZP-2012', 'Sapporo-DZP-2012-diffuse', 'Sapporo-DZP-diffuse', 'Sapporo-QZP', 'Sapporo-QZP-2012', 'Sapporo-QZP-2012-diffuse', 'Sapporo-QZP-diffuse', 'Sapporo-TZP', 'Sapporo-TZP-2012', 'Sapporo-TZP-2012-diffuse', 'Sapporo-TZP-diffuse', 'saug-ano-pV5Z', 'saug-ano-pVDZ', 'saug-ano-pVQZ', 'saug-ano-pVTZ', 'SBKJC Polarized (p,2d) - LFK', 'SBKJC-VDZ', 'SBO4-DZ(d)-3G', 'SBO4-DZ(d,p)-3G', 'SBO4-SZ-3G', 'Scaled MINI', 'sigmaDZHF', 'sigmaSZHF', 'sigmaTZHF', 'STO-2G', 'STO-3G', 'STO-4G', 'STO-5G', 'STO-6G', 'TZP-ZORA', 'UGBS', 'un-ccemd-ref', 'un-pcemd-ref', 'x2c-JFIT', 'x2c-JFIT-universal', 'x2c-QZVPall', 'x2c-QZVPall-2c', 'x2c-QZVPall-2c-s', 'x2c-QZVPall-s', 'x2c-QZVPPall', 'x2c-QZVPPall-2c', 'x2c-QZVPPall-2c-s', 'x2c-QZVPPall-s', 'x2c-SV(P)all', 'x2c-SV(P)all-2c', 'x2c-SV(P)all-s', 'x2c-SVPall', 'x2c-SVPall-2c', 'x2c-SVPall-s', 'x2c-TZVPall', 'x2c-TZVPall-2c', 'x2c-TZVPall-s', 'x2c-TZVPPall', 'x2c-TZVPPall-2c', 'x2c-TZVPPall-s'])
    st.write("*The listed Basis Set Exchange basis sets are available for C, H, O, N, F, S, and P. Future updates will extend this to include all atoms up to chlorine.*")

verbose_option = st.slider("Verbose", min_value=3, max_value=9, value=5)

#Second Input (NEW) - Pressure of the system
# pressure = 101325 #in Pascals (Pa), 101325 Pa = 1 atm
#Third Input (NEW) - Temperature of the system
# temperature = 298.15 #in K, 298.15K = room temperature (25 degrees Celsius) 
thermo_row = row(2)
temp = thermo_row.number_input("Temperature (K)", min_value=0.0, value=298.15)
press = thermo_row.number_input("Pressure (Pa)", min_value=0.0, value=101325.0)

with tabQM9Database:
    selectedMolecule = st.selectbox(
        'Search QM9 Molecule Database', precomputed_molecules, index=precomputed_molecules.index("methane:CH4"))
    if st.button('Add to Queue', use_container_width=True, key="qm9db"):
        if selectedMolecule:
            parseDatafile = open(
                "precomputed_molecules/" + selectedMolecule + ".geom.txt", "r").readlines()[4:]
            parseDatafile = "\n".join(parseDatafile[:-1])
            addToQueue(parseDatafile, basis_option)
        else:
            st.warning(
                "Please select a molecule using dropdown menu or inputting a text file.")
            
with tabCCCBDBDatabase:
    selectedMolecule = st.selectbox(
        'Search UI Molecule Database', precomputed_molecules, index=precomputed_molecules.index("methane:CH4"))
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
    uploaded_file = st.file_uploader("Upload a XYZ input", type=["xyz","txt"])
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
            tmpmol = Chem.AddHs(rdkit_mol)
            AllChem.EmbedMolecule(tmpmol)
            smiles = Chem.MolToSmiles(tmpmol)
            start = timeit.default_timer()
            data = compute_pyscf(
                atom, bse_pyscf, basis, verbose_option, method_option, temp, press)
            total_time = timeit.default_timer() - start
            
            # tdict = {"atom": atom, "basis_option": basis, "verbose_option": verbose_option, "temperature": temp, "pressure": press}
            # response = requests.post(api_url, params=tdict)
            
            # if response.status_code == 200:
            #     data = response.json()
            #     print("Yay, it worked!")
            # else:
            #     print(f"Error: {response.status_code} - {response.text}")   
            data['Atoms'] = rdkit_mol.GetNumAtoms()
            data['Bonds'] = rdkit_mol.GetNumBonds()
            data['Rings'] = rdkit_mol.GetRingInfo().NumRings()
            data['Weight'] = Descriptors.MolWt(rdkit_mol)
            data['Molecule'] = mol
            data['Rdkit Molecule'] = rdkit_mol
            data['Basis Source'] = bse_pyscf
            data['Basis'] = basis
            data['Molecule Name'] = getMoleculeName(atom)
            data['Smiles'] = smiles
            data['Real Compute Time'] = total_time
            st.session_state['counter'] += 1
            data['Run Order'] = st.session_state['counter']
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
        st.text("Total Real Runtime: " + str(round(sum(x['Real Compute Time'] for x in st.session_state['results']),2)) + "s")
        #st.text("Log Hessian Wall Runtime: " + str(round(sum(x['Hessian Wall Runtime'] for x in st.session_state['results']),2)) + "s")
        st.text(str(round(sum(x['Hessian Wall Runtime'] for x in st.session_state['results']),2)))
        st.text("Total Log CPU Runime: " + str(round(sum(x['SCF CPU Runtime'] + x['Hessian CPU Runtime'] for x in st.session_state['results']),2)) + "s")
        st.text("Total Log Wall Runtime: " + str(round(sum(x['SCF Wall Runtime'] + x['Hessian Wall Runtime'] for x in st.session_state['results']),2)) + "s")
        st.text("Log SCF Wall Runtime: " + str(round(sum(x['SCF Wall Runtime'] for x in st.session_state['results']),2)) + "s")
        
        
        for result_item in st.session_state['results']:
            data = result_item
            energy = {
                'Internal Energy (E - Ha)':[data['Internal Energy (at given T) (Ha)'],data['Electronic Internal Energy (Ha)'],data['Vibrational Internal Energy (Ha)'],data['Translational Internal Energy (Ha)'],data['Rotational Internal Energy (Ha)']],
                'Helmholtz Free Energy (F - Ha)':[data['Helmholtz Free Energy (Ha)'],data['Electronic Helmholtz Free Energy (Ha)'],data['Vibrational Helmholtz Free Energy (Ha)'],data['Translational Helmholtz Free Energy (Ha)'],data['Rotational Helmholtz Free Energy (Ha)']],
                'Gibbs Free Energy (G - Ha)':[data['Gibbs Free Energy (Ha)'],data['Electronic Gibbs Free Energy (Ha)'],data['Vibrational Gibbs Free Energy (Ha)'],data['Translational Gibbs Free Energy (Ha)'],data['Rotational Gibbs Free Energy (Ha)']],
                'Enthalpy (H - Ha)':[data['Enthalpy (Ha)'],data['Electronic Enthalpy (Ha)'],data['Vibrational Enthalpy (Ha)'],data['Translational Enthalpy (Ha)'],data['Rotational Enthalpy (Ha)']],  
            }
            pd.set_option("display.precision", 16)
            enerdf = pd.DataFrame(energy, index = ["Total","Electronic","Vibrational","Translational","Rotational"])
            
            entropy = {
                'Entropy (S - Ha/K)':[data['Entropy (Ha/K)'],data['Electronic Entropy (Ha/K)'],data['Vibrational Entropy (Ha/K)'],data['Translational Entropy (Ha/K)'],data['Rotational Entropy (Ha/K)']],
                'Helmholtz Free Entropy (Φ - Ha/K)':[data['Massieu Potential/Helmholtz Free Potential (Ha/K)'],data['Electronic Massieu Potential/Helmholtz Free Potential (Ha/K)'],data['Vibrational Massieu Potential/Helmholtz Free Potential (Ha/K)'],data['Translational Massieu Potential/Helmholtz Free Potential (Ha/K)'],data['Rotational Massieu Potential/Helmholtz Free Potential (Ha/K)']],
                'Gibbs Free Entropy (Ξ - Ha/K)':[data['Planck Potential/Gibbs Free Potential (Ha/K)'],data['Electronic Planck Potential/Gibbs Free Potential (Ha/K)'],data['Vibrational Planck Potential/Gibbs Free Potential (Ha/K)'],data['Translational Planck Potential/Gibbs Free Potential (Ha/K)'],data['Rotational Planck Potential/Gibbs Free Potential (Ha/K)']],
            }
            
            excluded_keys = ['Internal Energy (at given T) (Ha)', 'Electronic Internal Energy (Ha)', 'Vibrational Internal Energy (Ha)', 'Translational Internal Energy (Ha)', 'Rotational Internal Energy (Ha)', 'Helmholtz Free Energy (Ha)', 'Electronic Helmholtz Free Energy (Ha)', 'Vibrational Helmholtz Free Energy (Ha)', 'Translational Helmholtz Free Energy (Ha)', 'Rotational Helmholtz Free Energy (Ha)', 'Gibbs Free Energy (Ha)', 'Electronic Gibbs Free Energy (Ha)', 'Vibrational Gibbs Free Energy (Ha)', 'Translational Gibbs Free Energy (Ha)', 'Rotational Gibbs Free Energy (Ha)', 'Enthalpy (Ha)', 'Electronic Enthalpy (Ha)', 'Vibrational Enthalpy (Ha)', 'Translational Enthalpy (Ha)', 'Rotational Enthalpy (Ha)', 'Entropy (Ha/K)', 'Electronic Entropy (Ha/K)', 'Vibrational Entropy (Ha/K)', 'Translational Entropy (Ha/K)', 'Rotational Entropy (Ha/K)', 'Massieu Potential/Helmholtz Free Potential (Ha/K)', 'Electronic Massieu Potential/Helmholtz Free Potential (Ha/K)', 'Vibrational Massieu Potential/Helmholtz Free Potential (Ha/K)', 'Translational Massieu Potential/Helmholtz Free Potential (Ha/K)', 'Rotational Massieu Potential/Helmholtz Free Potential (Ha/K)', 'Planck Potential/Gibbs Free Potential (Ha/K)', 'Electronic Planck Potential/Gibbs Free Potential (Ha/K)', 'Vibrational Planck Potential/Gibbs Free Potential (Ha/K)', 'Translational Planck Potential/Gibbs Free Potential (Ha/K)', 'Rotational Planck Potential/Gibbs Free Potential (Ha/K)'] + ['Molecule', 'Rdkit Molecule', 'Basis', 'Molecule Name', 'Run Order','Atoms', 'Bonds', 'Rings', 'Weight', 'SCF CPU Runtime', 'SCF Wall Runtime', 'Hessian CPU Runtime', 'Hessian Wall Runtime', 'Basis Source']
            
            pd.set_option("display.precision", 16)
            entrodf = pd.DataFrame(entropy, index = ["Total","Electronic","Vibrational","Translational","Rotational"])
            
            with st.expander(str(st.session_state['results'].index(data) + 1) + "." + data['Molecule Name'] + " | "+data['Basis']+" (" + data['Basis Source']+" Basis): " + str(round(data['Real Compute Time'], 2)) + " s"):
                result_col_1, result_col_2 = st.columns([2, 1])
                result_col_1.write(f"SCF CPU Runtime: {data['SCF CPU Runtime']} s")
                result_col_1.write(f"SCF Wall Runtime: {data['SCF Wall Runtime']} s")
                result_col_1.write(f"Hessian CPU Runtime: {data['Hessian CPU Runtime']} s")
                result_col_1.write(f"Hessian Wall Runtime: {data['Hessian Wall Runtime']} s")
                result_col_1.write(
                    f"\# of Atoms: {data['Atoms']} | \# of Bonds: {data['Bonds']} | \# of Rings:  {data['Rings']}")
                result_col_1.write(
                    f"Molecular Weight: {data['Weight']} Da")
                # energy data
                for key, value in data.items():
                    if key not in excluded_keys:
                        result_col_1.write(f"{key}: {value}")

                with result_col_2:
                    speck_plot(
                        data['Molecule'], component_h=200, component_w=200, wbox_height="auto", wbox_width="auto")
                    st.image(MolToImage(data['Rdkit Molecule'], size=(200, 200)))
                    st.image(MolToImage(Chem.MolFromSmiles(data['Smiles']), size=(200, 200)))
                # linebreak
                st.write("")
                st.write("")
                
                
                col_config = {i:st.column_config.NumberColumn(i, format="%.4f") for i in enerdf.columns}
                st.dataframe(
                    data=enerdf, 
                    use_container_width=True,
                    column_config=col_config
                )
                col_config = {i:st.column_config.NumberColumn(i, format="%.4f") for i in entrodf.columns}
                st.dataframe(
                    data=entrodf, 
                    use_container_width=True,
                    column_config=col_config
                )
                

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
        
        exclude = [
            'Basis',
            'Rings',
            'Rdkit Molecule',
            'Converged SCF-HF Electronic Energy (Ha)',
            'Molecule',
            'Molecule Name',
            'Smiles',
            'Real Compute Time',
            'Run Order',
            'Basis Source',
        ]
        
        dependent = [i for i in st.session_state['results'][0].keys() if i not in independent]
        dependent = [i for i in dependent if i not in exclude]
        # print(dependent)
        
        df_columns = list(st.session_state['results'][0].keys())
        df_columns.remove('Rdkit Molecule')
        
        df = pd.DataFrame(st.session_state['results'], columns=df_columns)
        
        
        for label in independent:
            for target in dependent:
                # print(label, target)
                # print(df[label].values, df[target].values)
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
