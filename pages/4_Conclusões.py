import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Páginas da aplicação
st.set_page_config(
    page_title="Conclusões",
    page_icon="⭐️"
)
st.title("Conclusões")
st.sidebar.success("Escolha uma das páginas acima")

# Código de modelo

# importar os dados
DATA_PATH = "https://raw.githubusercontent.com/victor-ferreira/dataset/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(DATA_PATH)

# Alterando o estilo do seaborn para parecer com a do Streamlit
sns.set_style("darkgrid", {'text.color': 'white', 'xtick.color': 'white', 'ytick.color': 'white', 'axes.facecolor': '#0e1117', 'figure.facecolor': '#0e1117', 'axes.labelcolor': 'white'})
st.set_option('deprecation.showPyplotGlobalUse', False)

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

# Construindo e treinando o modelo
model = LogisticRegression()
model.fit(x_train, y_train)

# importar os dados
DATA_PATH = "https://raw.githubusercontent.com/victor-ferreira/dataset/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(DATA_PATH)


# Conclusões acerca dos modelos de previsão
st.subheader("Modelos de Previsão")
st.text("""
Todos os modelos tiveram resultados excelentes, porém, para conclusão de nossa 
pesquisa, utilizaremos os dados do modelo de Regressão Logística, cujo teve 82% 
de precisão e um ROC AUC de 73.5% auxiliado pelos clientes que foram registrados 
nesse dataset.

Tal resultado se trás muito útil, pois somente analisando a superfície do nosso
dataset tivemos uma previsão aproximada de 73% de acerto caso só chutassemos que os
clientes não cancelariam os serviços, sendo também mais alta do que uma estimativa
aleatória de 50-50. Isso significa que podemos efetivamente identificar clientes 
que estão prestes a sair com o nosso modelo, com uma alta precisão de 82%, trazendo
oportunidade para intervenção da empresa na decisão.

Também conseguimos identificar os principais fatores que fazem nossos clientes
sairem ou continuarem juntos da empresa. Aqui vai a lista dos principais fatores
que influenciam no churning:
""")
# Principais fatores de churning
st.subheader("Principais fatores de churning")

# Código

# Top 5 colunas que tem alta relação com o Churn
df2 = df.drop('customerID', axis=1)

# Convertendo strings para numéricos

for column in df2.columns:
  if df2[column].dtype == np.number:
    continue
  df2[column] = LabelEncoder().fit_transform(df2[column])

# Separando os dados
x = df2.drop('Churn', axis=1)
y = df2['Churn'].values.reshape(-1, 1)

# Mostrando o código que gerou
st.code("""
pesos = pd.Series(model.coef_[0], index=x.columns.values)
print(pesos.sort_values(ascending=False)[:5].plot(kind='barh'))
st.pyplot(plt.show())
""", "python")

# Gráfico dos serviços mais relacionados ao churning 

pesos = pd.Series(model.coef_[0], index=x.columns.values)
print(pesos.sort_values(ascending=False)[:5].plot(kind='barh', alpha=0.8))
st.pyplot(plt.show())

# Texto
st.text("""
Podemos ver que os 5 principais fatores que tendem a fazer as pessoas 
cancelarem os serviços são:

* Contrato Mensal: Pessoas que tem contrato mensal geralmente tendem a cancelar
mais os serviços do que clientes com contratos anuais.

* Serviço de Internet de fibra óptica: Clientes com fibra ótica tem mais chances de
fazer churning do que clientes sem os serviços de fibra ótica. 

* Pagamento no papel: Pessoas que fazem os pagamentos no papel tendem a ter
mais chances de cancelar os serviços.

* Total Alto de Dispesas: Pessoas com um total de dispesas altas.

* Múltiplas linhas de serviço: pessoas que não tem linhas telefônicas tendem a 
cancelar mais facilmente os serviços.
""")

# Motivos pelos quais os clientes permanecem 
st.subheader("Principais fatores de permanência:")

# Mostrando o código
st.code("""
pesos = pd.Series(model.coef_[0], index=x.columns.values)
print(pesos.sort_values(ascending=False)[-5:].plot(kind='barh'))
st.pyplot(plt.show())
""", "python")

# Gráfico dos serviços mais relacionados com a permanência
pesos = pd.Series(model.coef_[0], index=x.columns.values)
print(pesos.sort_values(ascending=False)[-5:].plot(kind='barh', alpha=0.8))
st.pyplot(plt.show())

# Texto
st.text("""
Podemos ver que os 5 principais fatores de permanência dos clientes são:

* Estar a muito tempo com a empresa: Clientes que já tem bastante tempo na 
empresa tendem a continuar com os serviços da mesma.

* Contratos anuais: Pessoas que tem contratos anuais, de 1 a 2 anos, tendem
a continuar junto com a empresa.

* Não tem várias linhas telefônicas: Pessoas que não possuem várias linhas
telefônicas são mais fiéis à ficar.

* Tem assinatura de segurança online: Possuir assinatura de segurança online
é um dos fatores mais ligados à fieldade com a empresa.

* Tem assinatura de suporte técnico: ter uma assinatura de Suporte técnico
faz com que os clientes continuem por mais tempo junto dos serviços.
""")

# Conclusão final
st.subheader("Conclusões do Projeto")
st.text("""Foi um projeto bastante divertido de ser feito, realmente bastante produtivo
e com uma infinita gama de possibilidades de uso, muito satisfatório conseguir
enxergar com outros olhos a facilidade que existe em analisar dados, 
corrigir-los, tratar-los para aplicarmos em modelos de aprendizagem. 

Confesso que sempre achei algo muito complicado só de olhar conteúdo online
sobre esses tópicos de machine learning e análise de dados, tenho muito a 
agradecer a Victor por ser um professor muito bom, assisti incontáveis vezes as 
aulas, com uma ajuda de um canal que também gosto chamado Corey Schafer, 
facilitaram bastante o entendimento de conceitos complexos e a prática dos
conteúdos vistos.

Espero poder fazer parte de mais projetos como esse e continuarei estudando
o assunto de ciência de dados pois é um tópico que realmente tenho apego e 
carinho por, muito obrigado se leu até aqui e até mais!
""")