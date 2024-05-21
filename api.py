from fastapi import FastAPI
from pyscf import gto, scf
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


app = FastAPI()

@app.post("/calculate")
def calculate(atom: str, basis_option: str, verbose_option: str, temperature: float, pressure: float):
    print(atom)
    print(basis_option)
    print(verbose_option)
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
    hessian_time = None
    energy = None
    for line in outputFile.readlines():
        if line.startswith("    CPU time for SCF"):
            time = float(line.split(" ")[-2])

        elif line.startswith("converged SCF energy = "):
            energy = float([i for i in line.split() if i != ''][4])
        elif line.startswith("    CPU time for UHF hessian"):
            hessian_time = float(line.split(" ")[-2])
    
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
        'Runtime': time,
        'Hessian Runtime': hessian_time,
        'Converged SCF-HF Nuclear Energy (Ha)': mf.energy_nuc(),
        'Converged SCF-HF Electronic Energy (Ha)': mf.energy_elec(),
        'Converged SCF-HF Total Energy (Ha)': mf.energy_tot(),
        # thermodynamic data
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

@app.post("/string-length")
def get_string_length(input_string: str):
    return {"length": len(input_string)}
    

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
