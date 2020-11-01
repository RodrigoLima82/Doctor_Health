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
import tensorflow as tf

from functions.utils import *
from functions.image_classification import *

from tensorflow.keras import backend as K 
import base64


st.set_option('deprecation.showfileUploaderEncoding', False)

def main():

    st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    activities = ["Home", "Stratifying Risks", "MRI Brain Tumor", "Breast Cancer", "Heart Disease", "Heart Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", activities)

    # ============================== HOME ======================================================= #
    if choice == "Home":

        st.header("Hello Folks! \nI am your Doctor Health and my goal is to offer the best experience for you.\nI already have some functions that can be used to you monitoring your health.")
        st.subheader("Features")
        st.write("- Risk Stratification Using Electronic Health Records")
        st.write("- Auto-Segmentation of Brain Tumor on Magnetic Resonance Imaging (MRI)")
        st.write("- Detection of Breast Cancer Injuries")
        st.write("- Heart Disease Prediction based on Age")
        st.write("- Heart Monitoring")

        image = PIL.Image.open("images/doctor-robot.png")
        st.image(image,caption="")


    # ============================== STRATIFYING RISKS ======================================================= #
    if choice == "Stratifying Risks":
        sub_activities = ["Predict"]
        sub_choice = st.sidebar.selectbox("Action", sub_activities)

        if sub_choice == "Predict":    

            df = getStratRiskFeatures()
            uploaded_file = False

            if st.checkbox('Want to upload data to predict?'):
                uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if st.button('Make predictions'):

                if uploaded_file:
                    df = pd.read_csv(uploaded_file, low_memory=False)
                    data = feature_engineering(df)
                    st.write(data)                
                else:                    
                    data = feature_engineering(df)
                    st.write(data)                

                ce       = joblib.load('fe_stratifying_risks/models/ce_leave.pkl')
                scaler   = joblib.load('fe_stratifying_risks/models/scaler.pkl')
                model    = joblib.load('fe_stratifying_risks/models/modelo_lr.pkl')

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
    
    # ============================== MRI BRAIN TUMOR ======================================================= #
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

    # ============================== BREAST CANCER ======================================================= #
    if choice == "Breast Cancer":
        sub_activities = ["Test", "Predict"]
        sub_choice = st.sidebar.selectbox("Action", sub_activities)

        if sub_choice == "Test":    

            LOCAL_MP4_FILE = "video/breastcancer.mp4"

            # play local video
            video_file = open(LOCAL_MP4_FILE, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

            st.write(" ")

            #uploaded_file = st.file_uploader("Choose a Breast image ...", type="png")
            #if uploaded_file is not None:
            #    image = PIL.Image.open(uploaded_file)
            #    st.image(image, caption='Uploaded Breast Image.',  width=100)
            #    st.write("")
            #    st.write("Classifying...")
            #    label = img_classification(image, 'fe_breast_cancer/model/modelo.h5')
            #    if label == 0:
            #        st.write("The image has NO invasive ductal carcinoma")
            #    else:
            #        st.write("The image has invasive ductal carcinoma")

            #st.write(" ")

    # ============================== HEART DISEASE ======================================================= #
    if choice == "Heart Disease":
        sub_activities = ["Predict"]
        sub_choice = st.sidebar.selectbox("Action", sub_activities)

        if sub_choice == "Predict":    

            LOCAL_MP4_FILE = "video/heartdisease.mp4"

            # play local video
            video_file = open(LOCAL_MP4_FILE, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

            st.write(" ")


            #df = getHeartDiseaseFeatures()
            #st.write(df)                
            #sex = df['sexo'][0]

            #if (sex == 0):
            #    img = PIL.Image.open("fe_heart_disease/images/maria.png")
            #else:
            #    img = PIL.Image.open("fe_heart_disease/images/bob.png")
            # 
            #st.image(img,caption="")


            #model = keras.models.load_model('fe_heart_disease/model/model_heart.h5')
            #X = np.asarray(df).astype(np.float32)
            #prediction = model.predict(X)
            #st.write("Heart disease can occur at the age of " + str(round(prediction[0][0],0)) + " years old")

    # ============================== HEART MONITOR ======================================================= #
    if choice == "Heart Monitor":
        sub_activities = ["Test", "Predict"]
        sub_choice = st.sidebar.selectbox("Action", sub_activities)

        if sub_choice == "Test":    

            LOCAL_MP4_FILE = "video/heartmonitoring.mp4"

            # play local video
            video_file = open(LOCAL_MP4_FILE, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

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