from Tkinter import *
import ttk
import pyscf
import threading
from pyscf import gto, scf

def pyscfThread():
    mol = gto.Mole()
    mol.atom = xyzInput.get()
    mol.basis = 'cc-pVTZ'
    mol.verbose = 5
    mol.output = 'output.txt'
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    

def calculate(*args):
    x = threading.Thread(target=pyscfThread)
    x.start()


root = Tk()
root.title("PySCF")

mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

xyzInput = StringVar()
xyzInput = ttk.Entry(mainframe, width=7, textvariable=xyzInput)
xyzInput.grid(column=1, row=1, sticky=(W, E))

energyOutput = StringVar()
ttk.Label(mainframe, textvariable=energyOutput).grid(column=1, row=2, sticky=(W, E))

ttk.Button(mainframe, text="Calculate", command=calculate).grid(
    column=1, row=3, sticky=W)

for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)

xyzInput.focus()
# root.bind("<Return>", calculate)

root.mainloop()


# mol = gto.Mole()
# mol.atom = """
# C	-0.4218040	0.6407370	0.0000000
# C	0.4218040	-0.6407370	0.0000000
# C	0.4218040	1.9198930	0.0000000
# C	-0.4218040	-1.9198930	0.0000000
# H	-1.0831630	0.6373520	0.8782670
# H	-1.0831630	0.6373520	-0.8782670
# H	1.0831630	-0.6373520	0.8782670
# H	1.0831630	-0.6373520	-0.8782670
# H	-0.2088070	2.8163860	0.0000000
# H	1.0686920	1.9672850	0.8847760
# H	1.0686920	1.9672850	-0.8847760
# H	0.2088070	-2.8163860	0.0000000
# H	-1.0686920	-1.9672850	0.8847760
# H	-1.0686920	-1.9672850	-0.8847760
# """
# mol.basis = 'cc-pVTZ'
# mol.verbose = 5
# mol.output = 'output.txt'
# mol.build()

# mf = scf.RHF(mol)
# mf.kernel()
