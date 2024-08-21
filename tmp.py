import os

for filename in os.listdir('./precomputed_molecules'):
    new_filename = filename.replace(':', '-')
    os.rename(os.path.join('./precomputed_molecules', filename), os.path.join('./precomputed_molecules', new_filename))