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

    activities = ["Home", "Stratifying Risks", "MRI Brain Tumor", "Breast Cancer", "Heart Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", activities)

    if choice == "Home":
        st.markdown("### ")
        st.write(" ")
        image = PIL.Image.open("images/image02.jpg")
        st.image(image,caption="", width=700)

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

            LOCAL_MP4_FILE = "video/breastcancer.mp4"

            # play local video
            video_file = open(LOCAL_MP4_FILE, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

            st.write(" ")

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