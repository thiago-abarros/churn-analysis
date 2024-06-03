# importar as bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
sns.set()

# Importando os métodos que serão usados

# Pré processamento de dados
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Treinamento e Construção dos modelos
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Métodos de Machine Learning
import tensorflow as tf
from tensorflow.python import keras 
from keras.src import Sequential
from keras.src.layers import Activation, Dense
from keras.src.optimizers import Adam
from keras.src.losses import categorical_crossentropy

# Ignorando os alertas
from warnings import simplefilter, filterwarnings
from sys import warnoptions
simplefilter(action='ignore', category=FutureWarning)
filterwarnings('ignore', category=FutureWarning)
if not warnoptions:
  simplefilter('ignore')
# importar os dados
DATA_PATH = "https://raw.githubusercontent.com/victor-ferreira/dataset/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(DATA_PATH)

# Páginas da aplicação
st.set_page_config(
    page_title="Modelos",
    page_icon="⭐️"
)

st.title("Modelos")
st.sidebar.success("Escolha uma das páginas acima")

# Alterando o estilo do seaborn para parecer com a do Streamlit
sns.set_style("darkgrid", {'text.color': 'white', 'xtick.color': 'white', 'ytick.color': 'white', 'axes.facecolor': '#0e1117', 'figure.facecolor': '#0e1117', 'axes.labelcolor': 'white'})
st.set_option('deprecation.showPyplotGlobalUse', False)

# Preparação dos dados

# Removendo colunas que não serão usadas:
df_limpo = df.drop('customerID', axis=1)

# Convertendo todas as colunas não numéricas para numéricas
for column in df_limpo.columns:
  if df_limpo[column].dtype == np.number:
    continue
  df_limpo[column] = LabelEncoder().fit_transform(df_limpo[column])

# Separando as variáveis independentes da variável dependente
x = df_limpo.drop('Churn', axis=1)
y = df_limpo['Churn']

# Transformando nossos dados para melhor performance dos modelos
x = StandardScaler().fit_transform(x)

# Separando os dados de treino e de teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Regressão Logística
st.subheader("Regressão Logística")

# Construindo e treinando o modelo
model = LogisticRegression()
model.fit(x_train, y_train)

# Fazendo a predição com o modelo
y_pred = model.predict(x_test)

# Texto:
st.code("""# Avaliando nosso modelo
print(classification_report(y_test, y_pred))""", "python")
st.code(""" # Retorno do print
              precision    recall  f1-score   support

           0       0.85      0.91      0.88      1036
           1       0.69      0.56      0.62       373

    accuracy                           0.82      1409
   macro avg       0.77      0.74      0.75      1409
weighted avg       0.81      0.82      0.81      1409
""", "python")

st.code("""# Vendo pela avaliação da curva ROC
print('ROC AUC: %.2f' % (roc_auc_score(y_test, y_pred)*100), '%')

# Visualizando o ROC
y_probs = model.predict_proba(x_test)[:,1]
fpr, tpr, threseholds = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, lw=2, color='blue')
plt.plot([0,1], [0,1], lw=2, color='gray')
plt.xlabel('Porcentagem de falsos positivos')
plt.ylabel('Porcentagem de verdadeiros positivos')
""", "python")

# Vendo pela avaliação da curva ROC

y_probs = model.predict_proba(x_test)[:,1]
fpr, tpr, threseholds = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, lw=2, color='blue')
plt.plot([0,1], [0,1], lw=2, color='gray')
plt.xlabel('Porcentagem de falsos positivos')
plt.ylabel('Porcentagem de verdadeiros positivos')
st.code("""ROC AUC: 73.52 %""")
st.pyplot(plt.show())


# GradientBoostingClassifier

# Código

# Construindo e Treinando o modelo
model = GradientBoostingClassifier(learning_rate=0.1, max_depth=2, n_estimators=200, max_features=8, random_state=42)
model.fit(x_train, y_train)

# Fazendo as predições
y_pred = model.predict(x_test)

# Texto
st.subheader("Gradient Boost Classifier")
st.code("""
# Construindo e Treinando o modelo
model = GradientBoostingClassifier(learning_rate=0.1, max_depth=2, 
n_estimators=200, max_features=8, random_state=42)
model.fit(x_train, y_train)

# Fazendo as predições
y_pred = model.predict(x_test)
""", "python")

st.code("""# Avaliando nosso modelo
print(classification_report(y_test, y_pred))
""", "python")
st.code("""# Retorno do print
              precision    recall  f1-score   support

           0       0.85      0.91      0.88      1036
           1       0.69      0.55      0.61       373

    accuracy                           0.82      1409
   macro avg       0.77      0.73      0.75      1409
weighted avg       0.81      0.82      0.81      1409
""", "python")

# Vendo pela avaliação da curva ROC

y_probs = model.predict_proba(x_test)[:,1]
fpr, tpr, threseholds = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, lw=2, color='blue')
plt.plot([0,1], [0,1], lw=2, color='gray')
plt.xlabel('Porcentagem de falsos positivos')
plt.ylabel('Porcentagem de verdadeiros positivos')
st.code("""ROC AUC: 73.04 %""")
st.pyplot(plt.show())


# Tensorflow Keras

# Texto
st.subheader("Tensorflow Keras")

st.code("""
# Criando o modelo de ML
model = Sequential([
Dense(20, input_shape=(19,), activation='relu'),
Dense(52, activation='relu'),
Dense(1, activation='sigmoid')
])

# Passando parâmetros de execução
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Fazendo o treinamento

model.fit(x_train, y_train, epochs=100)
""", "python")

st.code("""# Avaliando o modelo
pred_val = model.predict(x_test)
y_pred = []
for i in pred_val:
  if i > 0.5:
    y_pred.append(1)
  else:
    y_pred.append(0)

print(classification_report(y_test, y_pred))
""", "python")

st.code("""# Retorno do print
              precision    recall  f1-score   support

           0       0.83      0.88      0.85      1036
           1       0.60      0.48      0.53       373

    accuracy                           0.78      1409
   macro avg       0.71      0.68      0.69      1409
weighted avg       0.77      0.78      0.77      1409
""")


# XGBoost Classifier

# Código
# Construindo e Treinando o modelo
model = XGBClassifier(max_depth=1, learning_rate=0.3, n_estimators=300, seed=42)
model.fit(x_train, y_train)

# Fazendo as previsões
y_pred = model.predict(x_test)

# Texto
st.subheader("XGBoost Classifier")
st.code("""
# Construindo e Treinando o modelo
model = XGBClassifier(max_depth=1, learning_rate=0.3, n_estimators=300, seed=42)
model.fit(x_train, y_train)

# Fazendo as previsões
y_pred = model.predict(x_test)
""", "python")
st.code("""# Avaliando o model
print(classification_report(y_test, y_pred)) """, "python")
st.code("""# Retorno do print
              precision    recall  f1-score   support

           0       0.85      0.91      0.88      1036
           1       0.68      0.54      0.60       373

    accuracy                           0.81      1409
   macro avg       0.76      0.73      0.74      1409
weighted avg       0.80      0.81      0.80      1409
""", "python")
st.code("ROC AUC: 72.58 %", "python")

# Vendo pela avaliação da curva ROC
y_probs = model.predict_proba(x_test)[:,1]
fpr, tpr, threseholds = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, lw=2, color='blue')
plt.plot([0,1], [0,1], lw=2, color='gray')
plt.xlabel('Porcentagem de falsos positivos')
plt.ylabel('Porcentagem de verdadeiros positivos')
st.pyplot(plt.show())


# Naive Bayes Model - Gaussian

# Código
# Construindo e Treinando o modelo
model = GaussianNB()
model.fit(x_train, y_train)

# Fazendo as previsões
y_pred = model.predict(x_test)

# Texto
st.subheader("Naive Bayes Model - Gaussian")
st.code("""
# Construindo e Treinando o modelo
model = GaussianNB()
model.fit(x_train, y_train)

# Fazendo as previsões
y_pred = model.predict(x_test)
""", "python")
st.code("""# Avaliando o modelo
print(classification_report(y_test, y_pred))
""", "python")
st.code("""# Retorno do print
              precision    recall  f1-score   support

           0       0.90      0.75      0.82      1036
           1       0.53      0.77      0.63       373

    accuracy                           0.76      1409
   macro avg       0.72      0.76      0.72      1409
weighted avg       0.80      0.76      0.77      1409
""", "python")

# Vendo pela avaliação da curva ROC
y_probs = model.predict_proba(x_test)[:,1]
fpr, tpr, threseholds = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, lw=2, color='blue')
plt.plot([0,1], [0,1], lw=2, color='gray')
plt.xlabel('Porcentagem de falsos positivos')
plt.ylabel('Porcentagem de verdadeiros positivos')
st.code("ROC AUC: 76.25 %", "python")
st.pyplot(plt.show())

# Naive Bayes Model - Bernoulli

# Código
# Construindo e Treinando o modelo
model = BernoulliNB()
model.fit(x_train, y_train)

# Fazendo as previsões
y_pred = model.predict(x_test)

# Texto
st.subheader("Naive Bayes Model - Bernoulli")
st.code("""
# Construindo e Treinando o modelo
model = BernoulliNB()
model.fit(x_train, y_train)

# Fazendo as previsões
y_pred = model.predict(x_test)
""", "python")
st.code("""# Avaliando o modelo
print(classification_report(y_test, y_pred))
""", "python")
st.code("""# Retorno do print
              precision    recall  f1-score   support

           0       0.89      0.76      0.82      1036
           1       0.53      0.75      0.62       373

    accuracy                           0.76      1409
   macro avg       0.71      0.75      0.72      1409
weighted avg       0.80      0.76      0.77      1409
""", "python")
# Visualizando o ROC
y_probs = model.predict_proba(x_test)[:,1]
fpr, tpr, threseholds = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, lw=2, color='blue')
plt.plot([0,1], [0,1], lw=2, color='gray')
plt.xlabel('Porcentagem de falsos positivos')
plt.ylabel('Porcentagem de verdadeiros positivos')
st.code("ROC AUC: 75.43 %", "python")
st.pyplot(plt.show())
