# Importando as bibliotecas necessárias
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Páginas da aplicação
st.set_page_config(
    page_title="Inicio",
    page_icon="⭐️"
)
st.title("Página Principal")
st.sidebar.success("Escolha uma das páginas acima")

# Alterando o estilo do seaborn para parecer com a do Streamlit
sns.set_style("darkgrid", {'text.color': 'white', 'xtick.color': 'white', 'ytick.color': 'white',
              'axes.facecolor': '#0e1117', 'figure.facecolor': '#0e1117', 'axes.labelcolor': 'white'})
st.set_option('deprecation.showPyplotGlobalUse', False)

# Pegando o dataset e atribuindo ao DF
df = pd.read_csv(
    'https://raw.githubusercontent.com/victor-ferreira/dataset/main/WA_Fn-UseC_-Telco-Customer-Churn.csv')
st.header('Projeto de previsão de Churn')
st.text(""" 
O objetivo desta pesquisa é achar o melhor modelo para previsão do Churn de uma 
empresa, que basicamente é uma métrica que indica o quanto tal empresa perdeu 
de clientes.
Vamos analisar o dataset o qual iremos trabalhar: """)
df
# Tratamento dos dados

# Removendo colunas que não serão usadas:
df_limpo = df.drop('customerID', axis=1)

# Convertendo todas as colunas não numéricas para numéricas
for column in df_limpo.columns:
    if df_limpo[column].dtype == np.number:
        continue
    df_limpo[column] = LabelEncoder().fit_transform(df_limpo[column])

# Análises interativas
st.header("Aqui alguns valores da colunas numéricas")
# Barra de seleção de colunas
cols = ['MonthlyCharges', 'tenure', 'TotalCharges']
coluna_escolhida = st.selectbox(label='Selecione uma coluna', options=cols)

c1, c2, c3 = st.columns(3)
with c1:
    st.header('Mediana')
    st.metric(label='Mediana',
              value=df_limpo[coluna_escolhida].mean().__round__(1), delta=None)
with c2:
    st.header('Média')
    st.metric(label='Média',
              value=df_limpo[coluna_escolhida].median(),
              delta=((df_limpo[coluna_escolhida].mean() - df_limpo[coluna_escolhida].median()).__round__(4)))
with c3:
    st.header('Soma Geral')
    st.metric(label='Relação com o Churn',
              value=df_limpo[coluna_escolhida].sum().round(3))

# Mais alguns insights sobre essas duas variáveis
st.header('Principais colunas relacionadas com o Churn:')
tab1, tab2, tab3 = st.tabs(['Tenure', 'MonthlyCharges', 'InternetService'])

with tab1:
    st.subheader('Relação Churn e Tenure')
    st.bar_chart(data=df, x='Churn', y='tenure')

with tab2:
    st.subheader('Relação Churn e MonthlyCharges')
    st.bar_chart(data=df, x='Churn', y='MonthlyCharges')

with tab3:
    st.subheader('Relação Churn e InternetService')
    fig, ax = plt.subplots(figsize=(10.1, 4.865))
    sns.countplot(x='InternetService', hue='Churn', alpha=0.8, data=df)

    st.pyplot(plt.show())

# Relação Tenure, MonthlyCharges

st.subheader("Relação Churn com as respectivas colunas: ")
# Plotando gráficos

num_col = ['MonthlyCharges', 'tenure']
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
df[df.Churn == 'No'][num_col].hist(bins=20, color='#5988ac', alpha=0.8, ax=ax)
df[df.Churn == 'Yes'][num_col].hist(bins=20, color='#cc8963', alpha=0.8, ax=ax)
st.pyplot(plt.show())
st.code("""Laranja = Fez Churn
Azul = Não fez Churn""")

# Porcentagem de clientes que estão fazendo churn
st.subheader('Porcentagem dos clientes fazendo churn')

code = """# Código para calcular a porcentagem de Churn da empresa
print('Porcentagem de clientes que ainda permanecem com os serviços: ', 
df[df.Churn=='No'].shape[0]/(df[df.Churn=='Yes'].shape[0] + 
df[df.Churn=='No'].shape[0])*100)
print('Porcentagem de clientes que cancelaram os serviços: ',
df[df.Churn=='Yes'].shape[0]/(df[df.Churn=='Yes'].shape[0] + 
df[df.Churn=='No'].shape[0])*100)
"""
st.code(code, language='python')
st.code("""# Retorno do print
Porcentagem de clientes que ainda permanecem com os serviços:  73.4630129206304 
Porcentagem de clientes que cancelaram os serviços:  26.536987079369588""", "python")
st.text("""Saber disso é muito importante pois isso indica que nosso modelo precisará de mais
do que 73% de precisão para ser útil, visto que podemos fazer palpite de que os
clientes sempre ficarão inscritos nos nossos serviços que teremos no fim a mesma 
precisão.""")

# Dispersão dos dados
st.subheader("Visualizando dados de perfil de cliente:")

tab1, tab2, tab3, tab4 = st.tabs(
    ['gender', 'SeniorCitizen', 'Partner', 'Dependents'])

with tab1:
    st.subheader('Gênero:')
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(x=df['gender'], alpha=0.8, ax=ax)
    st.pyplot(plt.show())

with tab2:
    st.subheader('Idosos, 1 é idoso, 0 não é:')
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(x=df['SeniorCitizen'], alpha=0.8, ax=ax)
    st.pyplot(plt.show())

with tab3:
    st.subheader('Parceiros:')
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(x=df['Partner'], alpha=0.8, ax=ax)
    st.pyplot(plt.show())

with tab4:
    st.subheader('Se o cliente tem dependentes ou não: ')
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.countplot(x=df['Dependents'], alpha=0.8, ax=ax)
    st.pyplot(plt.show())
