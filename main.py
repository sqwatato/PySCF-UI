from time import sleep
import Tkinter as tk
import ScrolledText as st
import pyscf
import threading
from pyscf import gto, scf

queue = []

def pyscfThread():
    while (runThread):
        print(str(len(queue)) + " molecules in queue")
        if (len(queue) > 0):
            mol = gto.Mole()
            mol.atom = queue[0]
            queue.pop(0)
            print(mol.atom)
            sleep(3)
            # mol.basis = 'cc-pVTZ'
            # mol.verbose = 5
            # mol.output = 'output.txt'
            # mol.build()

            # mf = scf.RHF(mol)
            # mf.kernel()
            print("Completed computation")
        sleep(1)
    

def addToQueue():
    queue.append(xyzInput.get("1.0", tk.END))

def endThread():
    global runThread
    runThread = False
    root.destroy()




runThread = True
pythread = threading.Thread(target=pyscfThread)
pythread.start()

root = tk.Tk()
root.title("PySCF")
root.protocol("WM_DELETE_WINDOW", endThread)

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d" % (w, h))

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=4)

# Frames

xyzInputFrame = tk.Frame(root, background='red')
settingsFrame = tk.Frame(root, background='blue')
queueFrame = tk.Frame(root, background='green')
outputFrame = tk.Frame(root, background='black')
xyzInputFrame.pack_propagate(0)
settingsFrame.grid_propagate(0)
queueFrame.grid_propagate(0)
outputFrame.grid_propagate(0)

# XYZ Input
xyzInputFrame.rowconfigure(0, weight=3)
xyzInputFrame.rowconfigure(1, weight=1)

xyzInput = st.ScrolledText(xyzInputFrame)
xyzButton = tk.Button(xyzInputFrame, text="Compute", command=addToQueue)

xyzButton.pack(padx=4, pady=4, side="bottom", fill="both")
xyzInput.pack(padx=4, pady=4, side="top", fill="x")


xyzInputFrame.grid(column=0, row=0, sticky="nesw")
settingsFrame.grid(column=1, row=0, sticky="nesw")
queueFrame.grid(column=0, row=1, sticky="nesw")
outputFrame.grid(column=1, row=1, sticky="nesw")


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
