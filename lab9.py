import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

# Configura el tema estético del dashboard
st.set_page_config(page_title="Análisis de Tweets", layout="wide", initial_sidebar_state="expanded")

# Título principal
st.markdown("<h1 style='text-align: center; color: #274C77;'>Análisis de Tweets</h1>", unsafe_allow_html=True)

# Carga los datos
data_path = 'data/train.csv'
train_df = pd.read_csv(data_path)

# Preprocesamiento de texto con TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(train_df['text'].values)
y = train_df['target'].values

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenamiento de modelos
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
logreg_pred = logreg.predict(X_test)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# Sidebar para selección de modelos
st.sidebar.header("Modelos de Clasificación")
model_selection = st.sidebar.multiselect('Selecciona Modelos para Comparar', ['Logistic Regression', 'Decision Tree', 'Random Forest'])

# Visualización interactiva de resultados
st.sidebar.header("Exploración de Datos")
keyword_filter = st.sidebar.multiselect('Filtra por Palabra Clave', train_df['keyword'].unique())
target_filter = st.sidebar.selectbox('Filtra por Clasificación', [0, 1])

# Filtrar datos según selección del usuario
filtered_df = train_df[train_df['keyword'].isin(keyword_filter)] if keyword_filter else train_df
filtered_df = filtered_df[filtered_df['target'] == target_filter]

# Mostrar datos filtrados
st.write("### Datos Filtrados")
st.dataframe(filtered_df)

# Sección de comparación de modelos
if model_selection:
    st.write("### Resultados de Modelos Seleccionados")
    for model in model_selection:
        if model == 'Logistic Regression':
            st.write("#### Logistic Regression")
            st.write(confusion_matrix(y_test, logreg_pred))
            st.write(classification_report(y_test, logreg_pred))
        elif model == 'Decision Tree':
            st.write("#### Decision Tree")
            st.write(confusion_matrix(y_test, tree_pred))
            st.write(classification_report(y_test, tree_pred))
        elif model == 'Random Forest':
            st.write("#### Random Forest")
            st.write(confusion_matrix(y_test, rf_pred))
            st.write(classification_report(y_test, rf_pred))

# Gráficos: Distribución de la longitud del texto y su clasificación
st.write("### Distribución de Longitud de Tweets por Clasificación")
train_df['text_length'] = train_df['text'].str.len()
plt.figure(figsize=(10, 6))
sns.violinplot(x='target', y='text_length', data=train_df, palette=['#E7ECEF', '#274C77'])
st.pyplot(plt)

# Gráfico de dispersión para mostrar la longitud del tweet vs clasificación
st.write("### Relación entre Longitud del Tweet y Clasificación")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='text_length', y='target', data=train_df, hue='target', palette=['#6096BA', '#A3CEF1'])
st.pyplot(plt)

# Heatmap: Frecuencia de keywords
st.write("### Frecuencia de Keywords Principales")
keyword_counts = train_df['keyword'].value_counts().head(20)
plt.figure(figsize=(10, 8))
sns.heatmap(keyword_counts.to_frame().T, annot=True, cmap='Blues')
st.pyplot(plt)

# Tarjetas de estadísticas básicas
st.write("### Estadísticas Generales del Dataset")
col1, col2, col3 = st.columns(3)
col1.metric("Total Tweets", f"{len(train_df):,}")
col2.metric("Tweets de Desastres", f"{train_df['target'].sum():,}")
col3.metric("Tweets No Relacionados", f"{len(train_df) - train_df['target'].sum():,}")

# Estilo adicional
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #E7ECEF;
    }
    h1 {
        font-family: 'Helvetica', sans-serif;
        color: #274C77;
    }
    .stButton>button {
        background-color: #6096BA;
        color: white;
    }
</style>
""", unsafe_allow_html=True)
