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
from wordcloud import WordCloud

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

# Gráfico de barras para mostrar frecuencia de tweets sobre desastres
st.write("### Frecuencia de Tweets por Clasificación (Desastre vs No Desastre)")
plt.figure(figsize=(10, 6))
sns.countplot(x='target', data=train_df, palette=['#A3CEF1', '#274C77'])
plt.title("Frecuencia de Tweets Clasificados como Desastre vs No Desastre")
plt.xlabel("Clasificación")
plt.ylabel("Cantidad de Tweets")
st.pyplot(plt)

# Gráfico circular (pie chart) para proporción de tweets
st.write("### Proporción de Tweets por Clasificación")
labels = ['No Desastre', 'Desastre']
sizes = [train_df['target'].value_counts()[0], train_df['target'].value_counts()[1]]
colors = ['#A3CEF1', '#274C77']
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')  # Asegura que el pie chart sea un círculo
st.pyplot(plt)

# Nube de palabras para palabras clave más frecuentes asociadas a desastres
st.write("### Nube de Palabras de Keywords Relacionadas con Desastres")
disaster_keywords = train_df[train_df['target'] == 1]['keyword'].dropna().tolist()
disaster_text = ' '.join(disaster_keywords)
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(disaster_text)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Eliminar ejes para mejorar la visualización
st.pyplot(plt)

# Gráficos existentes:
# Gráficos de longitud de tweet por clasificación
st.write("### Distribución de Longitud de Tweets por Clasificación")
train_df['text_length'] = train_df['text'].str.len()
plt.figure(figsize=(10, 6))
sns.violinplot(x='target', y='text_length', data=train_df, palette=['#E7ECEF', '#274C77'])
st.pyplot(plt)

# Heatmap: Frecuencia de keywords
st.write("### Frecuencia de Keywords Principales")
keyword_counts = train_df['keyword'].value_counts().head(20)
plt.figure(figsize=(10, 8))
sns.heatmap(keyword_counts.to_frame().T, annot=True, cmap='Blues')
st.pyplot(plt)

# Crear la columna text_length si aún no existe
train_df['text_length'] = train_df['text'].str.len()

# Verificar que la columna text_length esté correctamente creada
st.write(train_df[['text', 'text_length']].head())  # Verifica que 'text_length' esté bien creada

# Gráfico de Caja (Boxplot) para Identificar Outliers
st.write("### Boxplot de Longitud de Tweets por Clasificación")
plt.figure(figsize=(10, 6))
sns.boxplot(x='target', y='text_length', data=train_df, palette=['#6096BA', '#A3CEF1'])
plt.title('Distribución de Longitud de Tweets por Clasificación')
plt.xlabel('Clasificación (0: No Desastre, 1: Desastre)')
plt.ylabel('Longitud del Tweet')
st.pyplot(plt)

# Gráfico de Densidad (KDE) para Longitud de Tweets
st.write("### Gráfico de Densidad (KDE) para Longitud de Tweets")
plt.figure(figsize=(10, 6))
sns.kdeplot(train_df[train_df['target'] == 0]['text_length'], label='No Desastre', shade=True, color='#A3CEF1')
sns.kdeplot(train_df[train_df['target'] == 1]['text_length'], label='Desastre', shade=True, color='#274C77')
plt.title("Densidad de Longitud de Tweets por Clasificación")
plt.xlabel('Longitud del Tweet')
plt.ylabel('Densidad')
plt.legend()
st.pyplot(plt)

# Matriz de Correlación de Características
st.write("### Matriz de Correlación de Características")
corr_matrix = train_df[['text_length', 'target']].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='Blues')
plt.title("Correlación entre Longitud de Texto y Clasificación")
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
