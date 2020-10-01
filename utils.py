import pandas as pd
import numpy as np
import re

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
