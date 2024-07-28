def getAtomicToMoleculeName():
    with open("elements.txt", "r") as f:
        lines = f.readlines()
        atomic_to_molecule_name = {}
        for line in lines:
            atomic_number, name, num = line.strip().split(",")
            atomic_to_molecule_name[atomic_number] = name
    return atomic_to_molecule_name

def getPyscfBasisSets():
    with open("pyscfbsets.txt", "r") as f:
        lines = [*map(str.strip, f.readlines())]
        return lines
    
def getBSEBasisSets():
    with open("bsebsets.txt", "r") as f:
        lines = [*map(str.strip, f.readlines())]
        return lines