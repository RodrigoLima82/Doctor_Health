import streamlit as st
import time
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from utils import *
from bokeh.models.widgets import Div
from sklearn.preprocessing import StandardScaler
import category_encoders as ce 

st.set_option('deprecation.showfileUploaderEncoding', False)

def main():

    html_page = """
    <div style="background-color:blue;padding=10px">
        <p style='color:white;text-align:center;font-size:20px;font-weight:bold'>DOCTOR HEALTH</p>
    </div>
              """
    st.markdown(html_page, unsafe_allow_html=True)    

    st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    activities = ["Home", "Predict","About"]
    choice = st.sidebar.selectbox("Menu",activities)

    if choice == "Home":
        st.markdown("### Stratifying risks using electronic records of diabetic patients")
        st.write("Segundo a Sociedade Brasileira de Diabetes (SBD) o Diabetes Mellitus (DM), ou simplesmente diabetes, é uma doença crônica na qual o corpo não produz insulina ou não consegue empregar adequadamente a insulina que produz. A insulina é um hormônio que controla a quantidade de glicose no sangue.")
        st.write("Segundo o Atlas, da Internacional Diabetes Federation (IDF), em 2019 o Brasil registrou 16,7 milhões de pessoas com diagnóstico de diabetes.")
        st.write(" ")
        image = Image.open("images/image01.png")
        st.image(image,caption="", width=700)


    if choice == "Predict":    

        race   = st.sidebar.selectbox("Race",("Caucasian","AfricanAmerican","Other","Asian","Hispanic"))
        gender = st.sidebar.radio("Gender",("Female","Male"), key = 'gender')
        age    = st.sidebar.selectbox("Age",("[0-10)","[10-20)","[20-30)","[30-40)","[40-50)","[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"))
        admission_type_id = st.sidebar.slider("Admission Type", 1,8,key='admission_type_id')
        discharge_disposition_id = st.sidebar.slider("Discharge Disposition", 1,30,key='discharge_disposition_id')
        admission_source_id = st.sidebar.slider("Admission Source", 1,26,key='admission_source_id')
        time_in_hospital = st.sidebar.number_input("Time in Hospital",1,100,step=1,key="time_in_hospital")
        num_lab_procedures = st.sidebar.number_input("Nro Lab Procedures",1,1000,step=1,key="num_lab_procedures")
        num_procedures = st.sidebar.number_input("Nro Procedures",1,1000,step=1,key="num_procedures")
        num_medications = st.sidebar.number_input("Nro Medications",1,1000,step=1,key="num_medications")
        number_outpatient = st.sidebar.number_input("Nro Out Patient",1,1000,step=1,key="number_outpatient")
        number_emergency = st.sidebar.number_input("Nro Emergency",1,1000,step=1,key="number_emergency")
        number_inpatient = st.sidebar.number_input("Nro Inpatient",1,1000,step=1,key="number_inpatient")

        diag_1 = st.sidebar.text_input('Diagnostic 1', value='250.83', key="diag_1")
        diag_2 = st.sidebar.text_input('Diagnostic 2', value='?', key="diag_2")
        diag_3 = st.sidebar.text_input('Diagnostic 3', value='?', key="diag_3")

        number_diagnoses = st.sidebar.number_input("Nro Diagnoses",1,100,step=1,key="number_diagnoses")

        max_glu_serum  = st.sidebar.selectbox("Max Glu Serum",(">200",">300","Norm","None"))
        A1Cresult  = st.sidebar.selectbox("A1C Result",(">7",">8","Norm","None"))


        #metformin   
        #repaglinide 
        #nateglinide 
        #chlorpropamide  
        #glimepiride acetohexamide   
        #glipizide   
        #glyburide   
        #tolbutamide 
        #pioglitazone    
        #rosiglitazone   
        #acarbose    
        #miglitol    
        #troglitazone    
        #tolazamide  
        #examide 
        #citoglipton 
        #insulin 
        #glyburide-metformin 
        #glipizide-metformin 
        #glimepiride-pioglitazone    
        #metformin-rosiglitazone 
        #metformin-pioglitazone  
        

        #change  
        #diabetesMed


        # Provide checkbox for uploading different training dataset
        if st.checkbox('Want to upload data to predict?'):
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if st.button('Make predictions'):

            result   = ""

            if uploaded_file:
                df = pd.read_csv(uploaded_file, low_memory=False)
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