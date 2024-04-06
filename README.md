(incomplete) UI for PySCF

- with streamlit installed, to run locally, run `streamlit run stream.py`
- `test.py` is for testing changes before changing `stream.py`
- put precomputed molecules in the `precomputed_molecules` folder
    - must end with `.geom.txt` and be formatted the same as the other files


### Full Steps
1. make python environment (if not already created): `python -m venv venv`
2. activate environment: `source venv/bin/activate`
3. install dependencies (if not already installed): `pip install -r requirements.txt`
4. run app: `streamlit run stream.py`
5. open link printed in terminal in web browser and use it!

Dependencies:
- in requirements.txt
- main ones
    - pyscf
    - rdkit
    - streamlit
    - altair
    - pandas
    - numpy
    - matplotlib
    - py3Dmol
