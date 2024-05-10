import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import subprocess

logo = Image.open("bch project logo placeholder.png")
st.image(logo)

st.write ("""
Drug Prediction App
          
By Onomeyimi Onesi
          
Dataset obtained from ChEMBL database

The goal of this project is to create a linear regression model that utilizes ChEMBL bioactivity data 
to generate inhibitor bioactivity predictions with respect to a specified target of interest. 
The test case shown here uses epidermal growth factor receptor (EGFR) as a target. 
This protein was selected as a target of interest due to its applications in cancer drug development.

""")

st.sidebar.header(("User Input"))

SMILES_input = "CCCCC\nCCCC\nCN"
chem_id = "A1"

SMILES = st.sidebar.text_area("SMILES input", SMILES_input)
SMILES = SMILES.strip()
SMILES = SMILES.replace(" ", "")
SMILES = SMILES.replace("\n", "")

st.header("Input SMILES")
SMILES

drug = [[chem_id, SMILES]]
df = pd.DataFrame(drug)
df.columns = ["chem_id", "canonical_smiles"]

# calculating molecular descriptors
def lipinski(smiles):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)

    i = 0
    baseData = np.array([])  # Initialize an empty array 

    for mol in moldata:
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
        
        row = np.array([desc_MolWt, desc_MolLogP, desc_NumHDonors, desc_NumHAcceptors]).reshape(1, -1)  # Reshape row to have 1 row and 4 columns

        if i == 0:
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
        
        i += 1

    columnNames = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
    descriptors = pd.DataFrame(data = baseData, columns = columnNames)

    return descriptors

df_lipinski = lipinski(df.canonical_smiles)

# combine the dataframes of the molecular descriptors and the original data
df2 = pd.concat([df, df_lipinski], axis = 1)

selection = ["canonical_smiles", "chem_id"]
df2_selection = df2[selection]
df2_selection.to_csv("molecule.smi", sep="\t", index = False, header = False)

subprocess.run("padel.sh", shell = True)

df3_X = pd.read_csv('descriptors_output.csv')
df3_X = df3_X.drop(columns=['Name'])

st.header("Compound's Computed Descriptors")
df3_X

# convert pIC50 to IC50 to determine the bioactivity class
def IC50(input):
    i = 10**(-input) * (10**9)

    return i

# Load the model
load_model = pickle.load(open("06model.pkl", "rb"))

st.header("Predicted Value")

if df3_X.shape == (0, 881):
    st.write("The predicted value for this compound could not be determined successfully :(. Try again.")
else:
    prediction = load_model.predict(df3_X)

    IC50pred = IC50(prediction[0])

    # assigning the bioactivity class
    bioactivity_threshold = []
        
    if float(IC50pred) >= 10000:
        bioactivity_threshold.append("inactive")
    elif float (IC50pred) <= 1000:
        bioactivity_threshold.append("active")
    else:
        bioactivity_threshold.append("intermediate")

    st.write("The compound {} is {} with a pIC50 value of {}".format(SMILES, bioactivity_threshold[0], prediction[0]))