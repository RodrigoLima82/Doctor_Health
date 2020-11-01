import streamlit as st
import time
import pandas as pd
import numpy as np
import joblib
import PIL
from bokeh.models.widgets import Div
from sklearn.preprocessing import StandardScaler
import category_encoders as ce 
import keras
import json
import nibabel as nib

from functions.utils import *
from functions.image_classification import *

from tensorflow.keras import backend as K 
import base64

st.set_option('deprecation.showfileUploaderEncoding', False)

def main():

    html_page = """
    <div style="background-color:blue;padding=10px">
        <p style='color:white;text-align:center;font-size:20px;font-weight:bold'>DOCTOR HEALTH</p>
    </div>
              """
    st.markdown(html_page, unsafe_allow_html=True)    

    st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    activities = ["Home", "Stratifying Risks", "MRI Brain Tumor", "Breast Cancer", "About"]
    choice = st.sidebar.selectbox("Menu", activities)

    if choice == "Home":
        st.markdown("### ")
        st.write(" ")
        image = PIL.Image.open("images/image02.jpg")
        st.image(image,caption="", width=700)

    if choice == "Stratifying Risks":
        sub_activities = ["Predict"]
        sub_choice = st.sidebar.selectbox("Stratifying Risks", sub_activities)

        if sub_choice == "Predict":    

            race   = st.sidebar.selectbox("Race",("Caucasian","AfricanAmerican","Other","Asian","Hispanic"))
            gender = st.sidebar.radio("Gender",("Female","Male"), key = 'gender')
            age    = st.sidebar.selectbox("Age",("[0-10)","[10-20)","[20-30)","[30-40)","[40-50)","[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"))
            admission_type_id = st.sidebar.slider("Admission Type", 1,8,key='admission_type_id')
            discharge_disposition_id = st.sidebar.slider("Discharge Disposition", 0,30,key='discharge_disposition_id')
            admission_source_id = st.sidebar.slider("Admission Source", 0,26,key='admission_source_id')
            time_in_hospital = st.sidebar.slider("Time in Hospital", 0,100,key='time_in_hospital')

            num_lab_procedures = st.sidebar.slider("Nro Lab Procedures",0,1000,step=1,key="num_lab_procedures")
            num_procedures = st.sidebar.slider("Nro Procedures",0,1000,step=1,key="num_procedures")
            num_medications = st.sidebar.slider("Nro Medications",0,1000,step=1,key="num_medications")
            number_outpatient = st.sidebar.slider("Nro Out Patient",0,1000,step=1,key="number_outpatient")
            number_emergency = st.sidebar.slider("Nro Emergency",0,1000,step=1,key="number_emergency")
            number_inpatient = st.sidebar.slider("Nro Inpatient",0,1000,step=1,key="number_inpatient")

            diag_1 = st.sidebar.text_input('Diagnostic 1', value='?', key="diag_1")
            diag_2 = st.sidebar.text_input('Diagnostic 2', value='?', key="diag_2")
            diag_3 = st.sidebar.text_input('Diagnostic 3', value='?', key="diag_3")

            number_diagnoses = st.sidebar.slider("Nro Diagnoses",0,100,step=1,key="number_diagnoses")

            max_glu_serum  = st.sidebar.selectbox("Max Glu Serum",(">200",">300","Norm","None"))
            A1Cresult  = st.sidebar.selectbox("A1C Result",(">7",">8","Norm","None"))

            metformin = st.sidebar.radio("Metformin",("No","Steady","Up","Down"),key='metformin')
            repaglinide = st.sidebar.radio("Repaglinide",("No","Steady","Up","Down"),key='repaglinide')
            nateglinide = st.sidebar.radio("Nateglinide",("No","Steady","Up","Down"),key='nateglinide')
            chlorpropamide  = st.sidebar.radio("Chlorpropamide",("No","Steady","Up","Down"),key='chlorpropamide')
            glimepiride = st.sidebar.radio("Glimepiride",("No","Steady","Up","Down"),key='glimepiride')
            acetohexamide  = st.sidebar.radio("Acetohexamide",("No","Steady","Up","Down"),key='acetohexamide')
            glipizide   = st.sidebar.radio("Glipizide",("No","Steady","Up","Down"),key='glipizide')
            glyburide   = st.sidebar.radio("Glyburide",("No","Steady","Up","Down"),key='glyburide')
            tolbutamide = st.sidebar.radio("Tolbutamide",("No","Steady","Up","Down"),key='tolbutamide')
            pioglitazone    = st.sidebar.radio("Pioglitazone",("No","Steady","Up","Down"),key='pioglitazone')
            rosiglitazone   = st.sidebar.radio("Rosiglitazone",("No","Steady","Up","Down"),key='rosiglitazone')
            acarbose    = st.sidebar.radio("Acarbose",("No","Steady","Up","Down"),key='acarbose')
            miglitol    = st.sidebar.radio("Miglitol",("No","Steady","Up","Down"),key='miglitol')
            troglitazone    = st.sidebar.radio("Troglitazone",("No","Steady","Up","Down"),key='troglitazone')
            tolazamide  = st.sidebar.radio("Tolazamide",("No","Steady","Up","Down"),key='tolazamide')
            examide = st.sidebar.radio("Examide",("No","Steady","Up","Down"),key='examide')
            citoglipton = st.sidebar.radio("Citoglipton",("No","Steady","Up","Down"),key='citoglipton')
            insulin = st.sidebar.radio("Insulin",("No","Steady","Up","Down"),key='insulin')
            glyburide_metformin = st.sidebar.radio("Glyburide Metformin",("No","Steady","Up","Down"),key='glyburide_metformin')
            glipizide_metformin = st.sidebar.radio("Glipizide Metformin",("No","Steady","Up","Down"),key='glipizide_metformin')
            glimepiride_pioglitazone    = st.sidebar.radio("Glimepiride",("No","Steady","Up","Down"),key='glimepiride_pioglitazone')
            metformin_rosiglitazone = st.sidebar.radio("Metformin Rosiglitazone",("No","Steady","Up","Down"),key='metformin_rosiglitazone')
            metformin_pioglitazone  = st.sidebar.radio("Metformin Pioglitazone",("No","Steady","Up","Down"),key='metformin_pioglitazone')
            
            change  = st.sidebar.radio("Change",("No","Ch"),key='change')
            diabetesMed = st.sidebar.radio("diabetesMed",("No","Yes"),key='diabetesMed')
            
            uploaded_file = False

            if st.checkbox('Want to upload data to predict?'):
                uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if st.button('Make predictions'):

                if uploaded_file:
                    df = pd.read_csv(uploaded_file, low_memory=False)
                    data = feature_engineering(df)
                    st.write(data)                
                else:
                    df = pd.DataFrame({'race': [race],
                                        'gender': [gender],
                                        'age': [age],
                                        'admission_type_id': [admission_type_id],
                                        'discharge_disposition_id': [discharge_disposition_id],
                                        'admission_source_id': [admission_source_id],
                                        'time_in_hospital': [time_in_hospital],
                                        'num_lab_procedures': [num_lab_procedures],
                                        'num_procedures': [num_procedures],
                                        'num_medications': [num_medications],
                                        'number_outpatient': [number_outpatient],
                                        'number_emergency': [number_emergency],
                                        'number_inpatient': [number_inpatient],
                                        'diag_1': [diag_1],
                                        'diag_2': [diag_2],
                                        'diag_3': [diag_3],
                                        'number_diagnoses': [number_diagnoses],
                                        'max_glu_serum': [max_glu_serum],
                                        'A1Cresult': [A1Cresult],
                                        'metformin': [metformin],
                                        'repaglinide': [repaglinide],
                                        'nateglinide': [nateglinide],
                                        'chlorpropamide': [chlorpropamide],
                                        'glimepiride': [glimepiride],
                                        'acetohexamide': [acetohexamide],
                                        'glipizide': [glipizide],
                                        'glyburide': [glyburide],
                                        'tolbutamide': [tolbutamide],
                                        'pioglitazone': [pioglitazone],
                                        'rosiglitazone': [rosiglitazone],
                                        'acarbose': [acarbose],
                                        'miglitol': [miglitol],
                                        'troglitazone': [troglitazone],
                                        'tolazamide': [tolazamide],
                                        'examide': [examide],
                                        'citoglipton': [citoglipton],
                                        'insulin': [insulin],
                                        'glyburide-metformin': [glyburide_metformin],
                                        'glipizide-metformin': [glipizide_metformin],
                                        'glimepiride-pioglitazone': [glimepiride_pioglitazone],
                                        'metformin-rosiglitazone': [metformin_rosiglitazone],
                                        'metformin-pioglitazone': [metformin_pioglitazone],
                                        'change': [change],
                                        'diabetesMed': [diabetesMed]})
                    data = feature_engineering(df)
                    st.write(data)                

                ce       = joblib.load('models/ce_leave.pkl')
                scaler   = joblib.load('models/scaler.pkl')
                model    = joblib.load('models/modelo_lr.pkl')

                data = ce.transform(data)
                data = scaler.transform(data)
                pred = model.predict_proba(data)[:,1] 
                
                # Dataframe com as probabilidades previstas (em dados de teste)
                df_proba = pd.DataFrame(pred, columns = ['Probabilidade'])

                # Dataframe para o risco estratificado
                df_risco = pd.DataFrame()

                # Agora carregamos o dataframe
                df_risco['Risco'] = df_proba.apply(classifica_risco, axis = 1)
                st.subheader("Predicted Readmission Risk for Patient")
                st.write("Risk: ", df_risco['Risco'][0])
    
    if choice == "MRI Brain Tumor":
        sub_activities = ["Test", "Predict"]
        sub_choice = st.sidebar.selectbox("Action", sub_activities)
        
        if sub_choice == "Test":    

            #model = load_model()
            image, label = load_case("fe_mri_brain_tumor/imagesTr/BRATS_003.nii.gz", "fe_mri_brain_tumor/labelsTr/BRATS_003.nii.gz")
            
            visualize_data_gif(get_labeled_image(image, label))

            """### Auto-segmentation of regions of a brain tumor using MRI"""
            file_ = open("/tmp/gif.gif", "rb")
            contents = file_.read()
            data_url = base64.b64encode(contents).decode("utf-8")
            file_.close()

            st.markdown(
                f'<img src="data:image/gif;base64,{data_url}" alt="teste">',
                unsafe_allow_html=True,
            )
            st.write(" ")            

    if choice == "Breast Cancer":
        sub_activities = ["Test", "Predict"]
        sub_choice = st.sidebar.selectbox("Action", sub_activities)

        if sub_choice == "Test":    

            uploaded_file = st.file_uploader("Choose a Breast image ...", type="png")
            if uploaded_file is not None:
                image = PIL.Image.open(uploaded_file)
                st.image(image, caption='Uploaded Breast Image.',  width=100)
                st.write("")
                st.write("Classifying...")
                label = img_classification(image, 'fe_breast_cancer/model/modelo.h5')
                if label == 0:
                    st.write("The image has NO invasive ductal carcinoma")
                else:
                    st.write("The image has invasive ductal carcinoma")
            
            st.write(" ")

    if choice == 'About':
        st.markdown("### Who I am")
        st.write(" - Hello Folks! I am your Doctor Health and my goal is to offer the best experience for you.")
        
        if st.button("website"):
            js = "window.open('https://www.linkedin.com/in/rodrigolima82/')"
            html = '<img src onerror="{}">'.format(js)
            div = Div(text=html)
            st.bokeh_chart(div)      

if __name__ == '__main__':
    main()