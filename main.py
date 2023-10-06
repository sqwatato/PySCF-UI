from time import sleep
import time
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
            print(mol.atom)
            sleep(3)
            mol.basis = 'cc-pVTZ'
            mol.verbose = 5
            # mol.output = 'computed/output_' + (str(time.time())) + '.txt'
            mol.output = 'computed/output-test.txt'
            mol.build()

            mf = scf.RHF(mol)
            mf.kernel()
            print("Completed computation")
            queue.pop(0)
            updateQueueText()
        sleep(1)
    
def getOutputThread():
    while (runThread):
        updateOutputText()
        sleep(0.2)

def addToQueue():
    queue.append(xyzInput.get("1.0", tk.END))
    updateQueueText()

def endThread():
    global runThread
    runThread = False
    root.destroy()

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

xyzInputFrame = tk.Frame(root, background='red', borderwidth=1, relief="solid")
settingsFrame = tk.Frame(root, background='blue', borderwidth=1, relief="solid")
queueFrame = tk.Frame(root, background='green', borderwidth=1, relief="solid")
outputFrame = tk.Frame(root, background='black', borderwidth=1, relief="solid")
xyzInputFrame.pack_propagate(0)
settingsFrame.grid_propagate(0)
queueFrame.pack_propagate(0)
outputFrame.pack_propagate(0)

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

queueText = tk.Text(queueFrame)

def updateQueueText():
    queueText.configure(state=tk.NORMAL)
    queueText.delete('1.0', tk.END)
    for queueItem in queue:
        queueText.insert(tk.END, queueItem.split('\n')[0])
        queueText.insert(tk.END, "\n---------------\n")
    queueText.configure(state=tk.DISABLED)

queueText.pack(expand=True,fill=tk.BOTH)
updateQueueText()

outputText = tk.Text(outputFrame)

currentText = ""
def updateOutputText():
    global currentText
    outputFile = open("computed/output-test.txt", "r")
    newText = outputFile.read()
    if (newText == currentText):
        return
    currentText = newText
    outputText.configure(state=tk.NORMAL)
    outputText.delete('1.0', tk.END)
    outputText.insert(tk.END, newText)
    outputText.see("end")
    outputText.configure(state=tk.DISABLED)

outputText.pack(expand=True,fill=tk.BOTH)
updateOutputText()


runThread = True
pythread = threading.Thread(target=pyscfThread)
pythread.start()
outputThread = threading.Thread(target=getOutputThread)
outputThread.start()

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
