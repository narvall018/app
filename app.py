#from logging import raiseExceptions
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from mordred import Calculator, descriptors
from rdkit.Chem import PandasTools
#import json
import joblib

import os
#import pybel
#from molmod import *
#from itertools import cycle
#from collections import namedtuple


###################
#FONCTIONS UTILES##
###################

# Compute descriptors
@st.cache
def mordred_calculator(esol_data):
    mordred_calc = Calculator(descriptors, ignore_3D=True)  # can't do 3D without sdf or mol filz
    mordred = mordred_calc.pandas([mol for mol in esol_data['ROMol']])
    return mordred

# Création d'un fonction permettant de lire XYZ et convertir au format SMILES
@st.cache
def xyz_to_smiles(fname: str) -> str:
    mol = next(pybel.readfile("xyz", fname))
    smi = mol.write(format="smi")
    return smi.split()[0].strip()


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Sélectionnez un fichier au format .xyz', filenames)
    return os.path.join(folder_path, selected_filename)



#########################
#MISE EN PAGE STREAMLIT##
#########################

def main():

    menu = ['Home',
    'Smiles conversion',
    'Database information',
    'Model performance information',
    "Prediction"]

    choice = st.sidebar.selectbox('Menu',menu)

    if choice == 'Home':
        #st.subheader('Acceuil')
        image = Image.open('img/VolatilityPredictorApp.png')
        st.image(image, use_column_width=True)
        st.write("""
# Volatility Predictor Web App
This application predicts **volatility ** values for organometallic compounds (boiling temperature and enthalpy of vaporization).  
The data to create the models were collected through different sources presented below:  
* [Chemsrc](https://www.chemsrc.com/en/Catg/134.html)  
* [Strem](https://www.strem.com/catalog/family/CVD+%26+ALD+Precursors/)
* [Nist] (https://www.nist.gov/)
* Chickos J.S., Acree W.E. ***Enthalpies of vaporization of organic and organometallic compounds *** , 1880–2002 // J. Phys.Chem. Ref. Data. – 2003. – Vol. 32. – Article No. 519.
""")    
        #Coordonnées
        expander_bar = st.beta_expander("My contact information")
        expander_bar.markdown("""
        * **Email:** juliensade75@gmail.com
        * **Linkedin:** [Profil] (https://www.linkedin.com/in/julien-sade-635b711ba/)
        * **Rapport:** Mettre le lien plus tard du rapport fini
        """)


    elif choice == "Smiles conversion":
        st.title('Smiles conversion')
        #st.subheader("Importez votre fichier")
        #xyz_file = st.file_uploader("Télécharger votre fichier xyz ou txt",
        #type =['txt',"xyz"])
        st.subheader('Convert xyz to SMILES')
        filename = st.text_input('Enter the path of your xyz/txt file:')
        if filename != '':
        	smi = xyz_to_smiles(filename)
        	st.write('The SMILES of your compound **%s**' % smi)

        st.subheader("Draw your molecule")
        st.write(""" You can draw your molecule on [Pubchem](https://pubchem.ncbi.nlm.nih.gov//edit3/index.html) to get the SMILES of your molecule
""")

        #filename = st.text_input('Enter a file path:')
        #smi = xyz_to_smiles(filename)
        #smi
        #st.title("Conversion")
        #st.subheader('Convertissez xyz en SMILES')
        #filename = file_selector()
        #st.write('Vous avez sélectionné `%s`' % filename)
        #smi = xyz_to_smiles(filename)
        #st.write('Le SMILES de votre composé **%s**' % smi)

        #if st.button('Convertir'):
            #if xyz_file is not None:
               # if xyz_file.type == "application/octet-stream" or "text/plain":
                    #Read docs
                    #raw_text = xyz_file.read()
                    #st.write(raw_text)
                    #Read as string
                    #raw_text = str(xyz_file.read(),"utf-8")








    elif choice == "Database information":
        st.header('Database information') 
        st.subheader("Number of occurrences for each metal")
        image = Image.open('img/metal_occurences.PNG')
        st.image(image, use_column_width=True)
        st.subheader("Number of occurrences for each type of ligand")
        image = Image.open('img/ligandtype_occurences.PNG')
        st.image(image, use_column_width=True)
        st.subheader("Histogram and density function of the predictor variable (Boiling temperature)")
        image = Image.open('img/densityV2.png')
        st.image(image, use_column_width=True)
        st.subheader("Whisker box of the predictor variable (Boiling temperature)")
        image = Image.open('img/boxplot.png')
        st.image(image, use_column_width=True)


    elif choice == "Model performance information":
        st.header('Model performance information') 
        st.subheader('Model performance')
        st.write('After comparing different models with statistical metrics, the model with the best performance after optimization is the XGBoost') 
        image = Image.open('img/PerformanceApresTuningEnSurlignage.PNG')
        st.image(image, use_column_width=True)
        st.subheader("Model learning curve")
        image = Image.open('img/LearningCurveV2.png')
        st.image(image, use_column_width=True)
        st.subheader("Prediction on the validation dataset")
        image = Image.open('img/PredictionAfterTuned.png')
        st.image(image, use_column_width=True)
        st.subheader("The most important variables of the model")
        image = Image.open('img/FeatureImportance.png')
        st.image(image, use_column_width=True)
        st.write('On the documentation of the [MORDRED] library (https://mordred-descriptor.github.io/documentation/master/descriptors.html), the different descriptors are described') 




    
    elif choice == "Prediction":
        st.title("Prediction")
        st.text("")
        st.text("")
        st.text("")
        ## Read SMILES input
        ## Read SMILES input
        SMILES_input = "[Li+].CCC[CH2-]\nC=CC[Sn](CC=C)(CC=C)CC=C"

        SMILES = st.sidebar.text_area("SMILES input", SMILES_input)
        SMILES = "C\n" + SMILES #Adds C as a dummy, first item
        SMILES = SMILES.split('\n')
        st.header('SMILES')

        SMILES_LIST = []
        for i in SMILES:
            SMILES_LIST.append(i)
   
        del SMILES_LIST[0]

        esol_data = pd.DataFrame(SMILES_LIST,columns=['smiles'])
        for index, row in esol_data.iterrows():
        	if row['smiles'] == "":
        		esol_data.drop(index, inplace=True)
        st.dataframe(esol_data)
        PandasTools.AddMoleculeColumnToFrame(esol_data, smilesCol='smiles')	
        for index, row in esol_data.iterrows():
        	if row['ROMol'] is None:
        		st.write(f"SMILES **{row[0]}** inserted is not recognized")
        		esol_data.drop(index, inplace=True)
        
        for index, row in esol_data.iterrows():
        	if row['ROMol'] is not None:
        		mordred = mordred_calculator(esol_data)
        		# remove non numerical features.
        		mordred = mordred.select_dtypes(include=['float64', 'int64', 'float'])
        		mordred = mordred.dropna(axis=1,how='all')
        		model_bp = joblib.load('mordred_modelBP.pkl')
        		prediction_bp = model_bp.predict(mordred)
        st.header('Prediction')
        esol_data = esol_data.drop("ROMol", axis=1)
        esol_data["prediction_boiling_point "] = prediction_bp
        st.dataframe(esol_data)
 


if __name__ == '__main__':
    main()
