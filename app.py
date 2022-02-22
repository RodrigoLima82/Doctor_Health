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

    activities = ["Home", "Stratifying Risks", "MRI Brain Tumor", "Breast Cancer", "Heart Disease", "Heart Risk", "Heart Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", activities)

    # ============================== HOME ======================================================= #
    if choice == "Home":

        st.header("Hello Folks! \nI am your Doctor Health and my goal is to offer the best experience for you.\nI already have some functions that can be used to you monitoring your health.")
        st.subheader("Features")
        st.write("- Risk Stratification Using Electronic Health Records")
        st.write("- Auto-Segmentation of Brain Tumor on Magnetic Resonance Imaging (MRI)")
        st.write("- Detection of Breast Cancer Injuries")
        st.write("- Heart Disease Prediction based on Age")
        st.write("- Cardiac Risk Assessment")
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
        sub_activities = ["Sample"]
        sub_choice = st.sidebar.selectbox("Action", sub_activities)
        
        if sub_choice == "Sample":    

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
        sub_activities = ["Predict", "Sample"]
        sub_choice = st.sidebar.selectbox("Action", sub_activities)

        if sub_choice == "Predict":    

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
            
        elif sub_choice == "Sample":    
            
            st.write("Video sample")
            LOCAL_MP4_FILE = "video/breastcancer.mp4"

            # play local video
            video_file = open(LOCAL_MP4_FILE, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

            st.write(" ")


    # ============================== HEART DISEASE ======================================================= #
    if choice == "Heart Disease":
        sub_activities = ["Predict", "Sample"]
        sub_choice = st.sidebar.selectbox("Action", sub_activities)

        if sub_choice == "Predict":    

            df = getHeartDiseaseFeatures()
            st.write(df)                
            sex = df['sexo'][0]

            if (sex == 0):
                img = PIL.Image.open("fe_heart_disease/images/maria.png")
            else:
                img = PIL.Image.open("fe_heart_disease/images/bob.png")
             
            st.image(img,caption="")

            model = keras.models.load_model('fe_heart_disease/model/model_heart.h5')
            X = np.asarray(df).astype(np.float32)
            prediction = model.predict(X)
            st.write("Heart disease can occur at the age of " + str(round(prediction[0][0],0)) + " years old")

            st.write(" ")
        
        elif sub_choice == "Sample":
            LOCAL_MP4_FILE = "video/heartdisease.mp4"

            # play local video
            video_file = open(LOCAL_MP4_FILE, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

            st.write(" ")


    # ============================== HEART RISK ======================================================= #
    if choice == "Heart Risk":
        sub_activities = ["Predict"]
        sub_choice = st.sidebar.selectbox("Action", sub_activities)
        
        if sub_choice == "Predict":    

            # Extrai o conteúdo do arquivo
            uploaded_file = False

            if st.checkbox('Want to upload data to predict?'):
                uploaded_file = st.file_uploader("Choose a MAT file", type="mat")

            if st.button('Make predictions'):

                if uploaded_file:
                    sinais = loadmat(uploaded_file)

                    sinais_mat = sinais['val']

                    # Extraindo o Batimento Cardiaco do Paciente
                    for channelid, channel in enumerate(sinais_mat):
                        resultado = ecg.ecg(signal = channel, sampling_rate = 300, show = False)
                        heart_rate = np.zeros_like(channel, dtype = 'float')
                        heart_rate = resultado['heart_rate']    
                        
                        st.write('BPM max: ', max(heart_rate).round(0))
                        
                        try:
                            if max(heart_rate) > 130:
                                HR = 1
                            else:
                                HR = 0
                        except:
                            continue    

                    # Carregamos o modelo   
                    modelo = load_model('fe_heart_sensor/model/ResNet_30s_34lay_16conv.hdf5')

                    # Valores constantes
                    frequencia = 300
                    tamanho_janela = 30 * frequencia

                    # Fazendo a previsao
                    x = processamento(sinais_mat, tamanho_janela)

                    # Previsões com o modelo (retorna as probabilidades)
                    prob_x, ann_x = previsoes(modelo, x)

                    # Realizando as previsoes
                    x = processamento(sinais_mat, tamanho_janela)
                    prob_x, ann_x = previsoes(modelo, x)
                    st.write('Probability FA (%): ', (prob_x[0, 0] * 100).round(2))

                    # Dataframe para o risco estratificado
                    df_risco = pd.DataFrame({'Probability':[prob_x[0, 0]], 'HR':HR})
                    df_risco['Risk'] = df_risco.apply(classifica_risco_cardiaco, axis = 1)
                    st.write('Risk: ', df_risco['Risk'][0])

                    # Plot
                    x_axis = np.linspace(0., float(len(sinais_mat[0]) / 300), num = len(sinais_mat[0]))
                    plt.rcParams.update({'font.size': 14})
                    fig, ax = plt.subplots(figsize = (16,5))

                    ax.plot(x_axis, sinais_mat[0], 'blue')
                    ax.axis([0, len(sinais_mat[0]) / 300, -2200, 2200])

                    ax.set_title('ECG Patient')
                    ax.set_xlabel("Time (seconds)")
                    ax.set_ylabel("Milli Volts")

                    st.write(fig)

    # ============================== HEART MONITOR ======================================================= #
    if choice == "Heart Monitor":
        sub_activities = ["Sample"]
        sub_choice = st.sidebar.selectbox("Action", sub_activities)

        if sub_choice == "Sample":    

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