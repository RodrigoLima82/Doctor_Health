import pandas as pd
import numpy as np
import re
import cv2
import imageio
import keras
import numpy as np
import nibabel as nib
import streamlit as st
import matplotlib.pyplot as plt
from IPython.display import Image
from keras import backend as K
from keras.layers import Input
from tensorflow.keras.models import Model
from keras.layers import Activation,Conv3D,MaxPooling3D,UpSampling3D
from tensorflow.keras.layers import Conv2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical
from tensorflow.compat.v1.logging import INFO, set_verbosity

set_verbosity(INFO)
K.set_image_data_format("channels_first")


# Função para preparar um dataframe com as feaures de Estratificacao de Riscos
def getStratRiskFeatures():

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

    return df


def calcula_comorbidade(row):
    
    # Código 250 indica diabetes
    codigos_doenca_diabetes = "^[2][5][0]"
    
    # Códigos 39x (x = valor entre 0 e 9)
    # Códigos 4zx (z = valor entre 0 e 6 e x = valor entre 0 e 9)
    # Esses códigos indicam problemas circulatórios
    codigos_doenca_circulatorios = "^[3][9][0-9]|^[4][0-5][0-9]"
    
    # Inicializa variável de retorno
    valor = 0
    
    # Valor 0 indica que:
    # Diabetes E problemas circulatórios não foram detectados de forma simultânea no paciente
    if(not(bool(re.match(codigos_doenca_diabetes, str(np.array(row['diag_1']))))) and
       not(bool(re.match(codigos_doenca_diabetes, str(np.array(row['diag_2']))))) and 
       not(bool(re.match(codigos_doenca_diabetes, str(np.array(row['diag_3'])))))) and (not(
        bool(re.match(codigos_doenca_circulatorios, str(np.array(row['diag_1']))))) and not(
        bool(re.match(codigos_doenca_circulatorios, str(np.array(row['diag_2']))))) and not(
        bool(re.match(codigos_doenca_circulatorios, str(np.array(row['diag_3'])))))):
        valor = 0
        
    # Valor 1 indica que:
    # Pelo menos um diagnóstico de diabetes E problemas circulatórios foram detectados de forma 
    # simultânea no paciente
    if(bool(re.match(codigos_doenca_diabetes, str(np.array(row['diag_1'])))) or 
       bool(re.match(codigos_doenca_diabetes, str(np.array(row['diag_2'])))) or 
       bool(re.match(codigos_doenca_diabetes, str(np.array(row['diag_3']))))) and (not(
        bool(re.match(codigos_doenca_circulatorios, str(np.array(row['diag_1']))))) and not(
        bool(re.match(codigos_doenca_circulatorios, str(np.array(row['diag_2']))))) and not(
        bool(re.match(codigos_doenca_circulatorios, str(np.array(row['diag_3'])))))): 
        valor = 1
        
    # Valor 2 indica que:
    # Diabetes E pelo menos um diagnóstico de problemas circulatórios foram detectados de forma 
    # simultânea no paciente
    if(not(bool(re.match(codigos_doenca_diabetes, str(np.array(row['diag_1']))))) and
       not(bool(re.match(codigos_doenca_diabetes, str(np.array(row['diag_2']))))) and 
       not(bool(re.match(codigos_doenca_diabetes, str(np.array(row['diag_3'])))))) and (
        bool(re.match(codigos_doenca_circulatorios, str(np.array(row['diag_1'])))) or 
        bool(re.match(codigos_doenca_circulatorios, str(np.array(row['diag_2'])))) or 
        bool(re.match(codigos_doenca_circulatorios, str(np.array(row['diag_3']))))):
        valor = 2
        
    # Valor 3 indica que:
    # Pelo menos um diagnóstico de diabetes e pelo menos um diagnóstico de problemas circulatórios 
    # foram detectados de forma simultânea no paciente
    if(bool(re.match(codigos_doenca_diabetes, str(np.array(row['diag_1'])))) or 
       bool(re.match(codigos_doenca_diabetes, str(np.array(row['diag_2'])))) or 
       bool(re.match(codigos_doenca_diabetes, str(np.array(row['diag_3']))))) and (
        bool(re.match(codigos_doenca_circulatorios, str(np.array(row['diag_1'])))) or 
        bool(re.match(codigos_doenca_circulatorios, str(np.array(row['diag_2'])))) or 
        bool(re.match(codigos_doenca_circulatorios, str(np.array(row['diag_3']))))):
        valor = 3 
    
    return valor

def feature_engineering(dados):

    # Recategorizamos 'idade' para que a população seja distribuída de maneira mais uniforme
    # Classificamos como faixa de 0-50 pacientes de até 50 anos
    dados['age'] = pd.Series(['[0-50)' if val in ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)'] else val 
                              for val in dados['age']], index = dados.index)

    # Acima de 80 anos ficam na faixa de 80-100
    dados['age'] = pd.Series(['[80-100)' if val in ['[80-90)', '[90-100)'] else val 
                              for val in dados['age']], index = dados.index)


    # A variável 'admission_type_id' contém 8 níveis
    # Reduziremos os níveis de 'admission_type_id' para duas categorias
    dados['admission_type_id'] = pd.Series(['Emergencia' if val == 1 else 'Outro' 
                                            for val in dados['admission_type_id']], index = dados.index)


    # A variável 'discharge_disposition_id' contém 26 níveis
    # Reduziremos os níveis de 'discharge_disposition_id' para duas categorias
    dados['discharge_disposition_id'] = pd.Series(['Casa' if val == 1 else 'Outro' 
                                                  for val in dados['discharge_disposition_id']], index = dados.index)


    # A variável 'admission_source_id' contém 17 níveis
    # # Reduziremos os níveis de 'admission_source_id' para três categorias
    dados['admission_source_id'] = pd.Series(['Sala_Emergencia' if val == 7 else 'Recomendacao' if val == 1 else 'Outro' 
                                                  for val in dados['admission_source_id']], index = dados.index)


    # Concatena 3 variáveis em um dataframe
    diagnostico = dados[['diag_1', 'diag_2', 'diag_3']]

    # Aplicamos a função comorbidade aos dados
    dados['comorbidade'] = diagnostico.apply(calcula_comorbidade, axis = 1)

    # Drop das variáveis individuais
    dados.drop(['diag_1','diag_2','diag_3'], axis = 1, inplace = True)
    
    # Removendo dataframe temporario
    del diagnostico

    # Lista com os nomes das variáveis de medicamentos
    medicamentos = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 
                    'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 
                    'troglitazone', 'tolazamide', 'examide', 'citoglipton','insulin', 'glyburide-metformin', 'glipizide-metformin', 
                    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']


    # Loop para ajustar o valor das variáveis de medicamentos
    for col in medicamentos:
        if col in dados.columns:
            colname = str(col) + 'temp'
            dados[colname] = dados[col].apply(lambda x: 0 if (x == 'No' or x == 'Steady') else 1)


    # Cria uma variável para receber a contagem por paciente
    dados['num_alt_dosagem_med'] = 0

    # Contagem de modificações na dosagem de medicamentos
    for col in medicamentos:
        if col in dados.columns:
            colname = str(col) + 'temp'
            dados['num_alt_dosagem_med'] = dados['num_alt_dosagem_med'] + dados[colname]
            del dados[colname]


    # Recoding das colunas de medicamentos
    for col in medicamentos:
        if col in dados.columns:
            dados[col] = dados[col].replace('No', 0)
            dados[col] = dados[col].replace('Steady', 1)
            dados[col] = dados[col].replace('Up', 1)
            dados[col] = dados[col].replace('Down', 1) 


    # Variável com a contagem de medicamentos por paciente
    dados['num_med'] = 0

    # Carregamos a nova variável
    for col in medicamentos:
        if col in dados.columns:
            dados['num_med'] = dados['num_med'] + dados[col]


    # Remove as colunas de medicamentos
    dados = dados.drop(columns = medicamentos)


    # Recoding de variáveis categóricas binárias
    dados['change'] = dados['change'].replace('Ch', 1)
    dados['change'] = dados['change'].replace('No', 0)
    dados['gender'] = dados['gender'].replace('Male', 1)
    dados['gender'] = dados['gender'].replace('Female', 0)
    dados['diabetesMed'] = dados['diabetesMed'].replace('Yes', 1)
    dados['diabetesMed'] = dados['diabetesMed'].replace('No', 0)


    # Recoding de variáveis categóricas (label encoding)
    dados['A1Cresult'] = dados['A1Cresult'].replace('>7', 1)
    dados['A1Cresult'] = dados['A1Cresult'].replace('>8', 1)
    dados['A1Cresult'] = dados['A1Cresult'].replace('Norm', 0)
    dados['A1Cresult'] = dados['A1Cresult'].replace('None', -99)
    dados['max_glu_serum'] = dados['max_glu_serum'].replace('>200', 1)
    dados['max_glu_serum'] = dados['max_glu_serum'].replace('>300', 1)
    dados['max_glu_serum'] = dados['max_glu_serum'].replace('Norm', 0)
    dados['max_glu_serum'] = dados['max_glu_serum'].replace('None', -99)


    return dados

# Função para estratificar o risco
def classifica_risco(row):
    if row[0] <= 0.3 : return 'Low'
    if row[0] >= 0.7 : return 'High'
    return 'Median'

def load_case(image_nifty_file, label_nifty_file):
    image = np.array(nib.load(image_nifty_file).get_fdata())
    label = np.array(nib.load(label_nifty_file).get_fdata())    
    return image, label

def visualize_data_gif(data_):
    images = []
    for i in range(data_.shape[0]):
        x = data_[min(i, data_.shape[0] - 1), :, :]
        y = data_[:, min(i, data_.shape[1] - 1), :]
        z = data_[:, :, min(i, data_.shape[2] - 1)]
        img = np.concatenate((x, y, z), axis=1)
        images.append(img)
    imageio.mimsave("/tmp/gif.gif", images, duration=0.01)
    return Image(filename="/tmp/gif.gif", format='png')

def get_labeled_image(image, label, is_categorical=False):
    if not is_categorical:
        label = to_categorical(label, num_classes=4).astype(np.uint8)

    image = cv2.normalize(image[:, :, :, 0], None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

    labeled_image = np.zeros_like(label[:, :, :, 1:])

    # remove tumor part from image
    labeled_image[:, :, :, 0] = image * (label[:, :, :, 0])
    labeled_image[:, :, :, 1] = image * (label[:, :, :, 0])
    labeled_image[:, :, :, 2] = image * (label[:, :, :, 0])

    # color labels
    labeled_image += label[:, :, :, 1:] * 255
    return labeled_image