import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Páginas da aplicação
st.set_page_config(
    page_title="Tratamento dos dados",
    page_icon="⭐️"
)
st.title("Tratamento dos dados")
st.sidebar.success("Escolha uma das páginas acima")
# estilo dos gráficos do  seaborn
sns.set_style("darkgrid", {'text.color': 'white', 'xtick.color': 'white', 'ytick.color': 'white', 'axes.facecolor': '#0e1117', 'figure.facecolor': '#0e1117', 'axes.labelcolor': 'white'})
# Importando os dados e criando o dataframe
df = pd.read_csv(
    'https://raw.githubusercontent.com/victor-ferreira/dataset/main/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Fazendo o tratamento dos dados do df_limpo

# Removendo colunas que não serão usadas:
df_limpo = df.drop('customerID', axis=1)
# Convertendo todas as colunas não numéricas para numéricas
for column in df_limpo.columns:
  if df_limpo[column].dtype == np.number:
    continue
  df_limpo[column] = LabelEncoder().fit_transform(df_limpo[column])

# Início

st.text("""
Fazer um tratamento de dados significa basicamente melhorararmos as formas de 
nossos dados para que os modelos preditivos entendam melhor a importância de 
cada atributo de nossa base de dados.""")
st.text("""
Temos que seguir uma série de passos para o correto tratamento de dados, que
é a análise exploratória dos dados, depois a conversão para valores numéricos (One 
Hot Encoding) e finalmente dividirmos os dados para treino e teste. """)

# Análise dos Atributos do Dataset

st.header("Análise dos Atributos do Dataset")
st.text("""
Os principais métodos que utilizaremos durante essa análise são os métodos info(), 
isnull(), duplicated() e value_counts(), que serão explicados mais profundamente 
nos exemplos abaixo: """)

# Método info()

st.subheader("Método info():")
st.text("""O método info(), como o nome diz, mostra as informações gerais do nosso Dataframe 
(basicamente nosso complexo de dados), tais informações que são retornadas são: 
* A quantidade de valores vazíos de cada coluna
* O tipo de dado que está armazenado em cada coluna
* O nome e quantidade de valores de cada coluna
""")

st.code("df.info()", "python")
st.code("""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7043 entries, 0 to 7042
Data columns (total 21 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   customerID        7043 non-null   object 
 1   gender            7043 non-null   object 
 2   SeniorCitizen     7043 non-null   int64  
 3   Partner           7043 non-null   object 
 4   Dependents        7043 non-null   object 
 5   tenure            7043 non-null   int64  
 6   PhoneService      7043 non-null   object 
 7   MultipleLines     7043 non-null   object 
 8   InternetService   7043 non-null   object 
 9   OnlineSecurity    7043 non-null   object 
 10  OnlineBackup      7043 non-null   object 
 11  DeviceProtection  7043 non-null   object 
 12  TechSupport       7043 non-null   object 
 13  StreamingTV       7043 non-null   object 
 14  StreamingMovies   7043 non-null   object 
 15  Contract          7043 non-null   object 
 16  PaperlessBilling  7043 non-null   object 
 17  PaymentMethod     7043 non-null   object 
 18  MonthlyCharges    7043 non-null   float64
 19  TotalCharges      7043 non-null   object 
 20  Churn             7043 non-null   object 
dtypes: float64(1), int64(2), object(18)
memory usage: 1.1+ MB
""", "python")

# Método isnull()

st.subheader("Método isnull():")
st.text("""O método isnull(), retorna quantos números nulos que há em cada coluna.""")
st.code("df.isnull().sum()", "python")
st.code("""
customerID          0
gender              0
SeniorCitizen       0
Partner             0
Dependents          0
tenure              0
PhoneService        0
MultipleLines       0
InternetService     0
OnlineSecurity      0
OnlineBackup        0
DeviceProtection    0
TechSupport         0
StreamingTV         0
StreamingMovies     0
Contract            0
PaperlessBilling    0
PaymentMethod       0
MonthlyCharges      0
TotalCharges        0
Churn               0
dtype: int64
""", "python")

# Método duplicated()

st.subheader("Método duplicated():")
st.text("""O método duplicated() checa se há duplicatas nas colunas.""")
st.code("df.duplicated().sum()", "python")
st.code("0", "python")

# Método value_counts()

st.subheader("Método value_counts():")
st.text("""O método value_counts() basicamente conta a quantidade da valores únicos de
cada coluna, podemos usar ele para plotar um gráfico com a comparação da quantidade
de valores únicos.""")
st.code("""
fig, ax = plt.subplots(figsize=(10.1, 4.865))
sns.countplot(x=df['Churn'], ax=ax)
st.pyplot(plt.show())
""", "python")

# Plot da figura

fig, ax = plt.subplots(figsize=(10.1, 4.865))
sns.countplot(x=df['Churn'], alpha=0.8, ax=ax)
st.pyplot(plt.show())

# Tratamento dos Dados

st.header("Tratamento dos Dados")
st.text("""
Agora que já sabemos os principais atributos do nosso Dataframe, 
vamos tratar os dados que estão nele para que sejam mais facilmente 
compreendidos pelos nossos modelos de previsão.""")

# Excluindo colunas

st.subheader("Excluindo colunas")
st.text("""
Vamos excluir as colunas que não são úteis para nossos cálculos de 
previsão, sendo a única coluna inútil a de "CustomerID", que 
somente guarda o ID único de cada cliente registrado. Faremos isso 
usando o método drop, utilizando o axis=1 porque queremos excluir 
toda a coluna.
""")
st.code("""
# Removendo colunas que não serão usadas:
df_limpo = df.drop('customerID', axis=1)
""", "python")

# Convertendo todas as colunas não numéricas para numéricas

st.subheader("Convertendo todas as colunas não numéricas para numéricas")
st.text("""
Vamos transformar todas as colunas com valores não numéricos para 
valores numéricos, porque os modelos de previsão compreendem muito 
melhor números do que caractéres. Podemos automatizar esse processo 
utilizando o método LabelEncoder da biblioteca sklearn voltada a pré
processamento. 

O LabelEncoder basicamente automatiza todo o processo
de One Hot Encoding, que é um método que transforma valores únicos 
de caractéres em tabelas numéricas que faz a mesma identificação.
""")
st.code("""
# Convertendo todas as colunas não numéricas para numéricas
for column in df_limpo.columns:
  if df_limpo[column].dtype == np.number:
    continue
  df_limpo[column] = LabelEncoder().fit_transform(df_limpo[column])
""", "python")

st.code("""
# Checando se foi bem sucedida a transformação
df_limpo.dtypes
""", "python")

st.code("""
gender                int64
SeniorCitizen         int64
Partner               int64
Dependents            int64
tenure                int64
PhoneService          int64
MultipleLines         int64
InternetService       int64
OnlineSecurity        int64
OnlineBackup          int64
DeviceProtection      int64
TechSupport           int64
StreamingTV           int64
StreamingMovies       int64
Contract              int64
PaperlessBilling      int64
PaymentMethod         int64
MonthlyCharges      float64
TotalCharges          int64
Churn                 int64
dtype: object
""", "python")

# Preparando os dados para os modelos

st.subheader("Preparando os dados para os modelos")
st.text("""
Vamos agora separar a nossa variável dependente das variáveis independentes, 
que basicamente é separar a coluna alvo que queremos prever, sendo essa a coluna 
Churn, das variáveis independentes, que são os parâmetros que nos ajudarão a 
fazer a previsão. 

Também vamos usar o StandardScaler para padronizar nossos dados para não ter
tanta dispersão de valores nos nossos dados, ajudando os modelos a reconhecer
os padrões de dados que iremos passar para os mesmos. 
""")
st.code("""
# Separando as variáveis independentes da variável dependente
x = df_limpo.drop('Churn', axis=1)
y = df_limpo['Churn']

# Transformando nossos dados para melhor performance dos modelos
x = StandardScaler().fit_transform(x)

""", "python")

st.text("""
Agora só precisamos separar nossos dados em dados de treino e dados de teste, 
que basicamente, é a separação de dados para treinar o modelo, o qual geralmente 
é a maior parte dos dados, e outra parte para treino, que serão os dados usados 
para verificar a precisão de nossos dados, sendo a menor parte dos dados.
""")
st.text("""
Faremos isso utilizando um método que já faz essa separação, que é o 
train_test_split, da biblioteca sklearn, que recebe as variáveis que serão
utilizadas (geralmente sendo x e y, x para as variáveis independentes e 
y para a variável alvo):
""")
st.code("""
# Separando os dados de treino e de teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
random_state=42)""", "python")

# Verificando a distribuição
st.text("""
Podemos verificar se as normalizações que fizemos funcionaram plotando o gráfico
com a linha de disparidade entre valores, utilizando o distplot() do seaborn.
""")

tab1, tab2, tab3 = st.tabs(['Tenure', 'MonthlyCharges', 'TotalCharges'])

with tab1:
    st.subheader('Relação de tempo de cliente junto aos serviços ')
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.distplot(df_limpo['tenure'])
    st.pyplot(plt.tight_layout())

with tab2:
    st.subheader('Relação de dispesas mensais')
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.distplot(df_limpo['MonthlyCharges'])
    st.pyplot(plt.tight_layout())

with tab3:
    st.subheader('Relação de custos totais')
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.distplot(df_limpo['TotalCharges'])
    st.pyplot(plt.tight_layout())
