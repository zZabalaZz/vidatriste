import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier  # Importamos RandomForestClassifier para la predicción

# Cargar los modelos y codificadores
@st.cache_data
def load_data():
    df = pd.read_csv('Delito Bucaramanga_preprocesar.csv', delimiter=",")  # Currently on my local machine
    return df

@st.cache_resource
def load_models():
    codEdad = joblib.load('codEdad.bin')
    codHorario = joblib.load('codHorario.bin')
    codGenero = joblib.load('codGenero.bin')
    codDia = joblib.load('codDia.bin')
    codComuna = joblib.load('codComuna.bin')
    modeloBA = joblib.load('modeloBA.bin')  # Cargamos el modelo RandomForestClassifier
    return codEdad, codHorario, codGenero, codDia, codComuna, modeloBA

# Cargar los datos y los modelos
df = load_data()
codEdad, codHorario, codGenero, codDia, codComuna, modeloBA = load_models()

#Configuración de la página
#st.set_page_config(
 #   page_title="Predicción de Delitos en Bucaramanga",
  #  page_icon="laley.jpeg",  # Cambia este archivo por el nombre o URL del icono que quieras usar
   # layout="centered"
#)

# Título y descripción del proyecto
st.markdown("<h1 style='color: #340467;'>Predicción de delitos en Bucaramanga</h1>", unsafe_allow_html=True)
st.markdown("---")
st.write("Brayan León - Sergio Amaya")

st.markdown("""
### Objetivo del Proyecto:
El propósito de este proyecto es determinar la probabilidad de ocurrencia de un delito en función de varias características observables.
El modelo predice aspectos clave como el barrio donde ocurre el delito, el tipo de delito, el móvil del agresor y el arma potencialmente usada.
Estas predicciones se basan en datos sobre la comuna, el rango de horario, el sexo y el curso de vida de la víctima, proporcionando una herramienta
que puede ayudar a las autoridades y a la comunidad en la prevención y análisis de patrones delictivos en Bucaramanga.
""")
st.markdown("---")

# Mostrar resumen de los datos
st.write("El número total de registros cargados es: ", len(df))
st.write("El rango de fechas de los delitos va desde ", pd.to_datetime(df['fecha_hecho']).min(), " hasta ", pd.to_datetime(df['fecha_hecho']).max())
st.write(f"El número de tipos de delitos registrados es: {len(df['delito_solo'].unique())}, distribuidos en {len(df['barrios_hecho'].unique())} barrios de {len(df['nom_com'].unique())} comunas.")

# Mostrar vista previa de los datos
st.subheader("Vista Previa de los Datos")
st.dataframe(df.head(5))

# Opciones del sidebar para entradas del usuario
st.sidebar.header("*Predicción Delitos Bucaramanga*")
st.sidebar.image("head.png", caption="", width=200)
st.sidebar.header("Características de entrada")

comunas = ['SUR', 'PROVENZA', 'LA CIUDADELA', 'LA CONCORDIA', 'CENTRO',
           'CABECERA DEL LLANO', 'ORIENTAL', 'OCCIDENTAL', 'SAN FRANCISCO',
           'NORORIENTAL', 'NORTE', 'SUROCCIDENTE', 'GARCIA ROVIRA',
           'MORRORICO', 'LA PEDREGOSA', 'MUTIS', 'LAGOS DEL CACIQUE']

meses = list(range(1, 13))
sexo_victima = ['MASCULINO', 'FEMENINO']
movil_victima = ['MOTOCICLETA', 'TAXI', 'VEHICULO', 'BUS', 'METRO']
dias_semana = ['LUNES', 'MARTES', 'MIERCOLES', 'JUEVES', 'VIERNES', 'SABADO', 'DOMINGO']
rango_horario = ['MADRUGADA', 'MAÑANA', 'NOCHE', 'TARDE']
rango_edad = ['ADOLESCENCIA', 'ADULTEZ', 'INFANCIA', 'JUVENTUD', 'PERSONA MAYOR', 'PRIMERA INFANCIA']

# Selectores en el sidebar
comuna = st.sidebar.selectbox("Seleccione la comuna", comunas)
mes = st.sidebar.selectbox("Seleccione el mes", meses)
sexo = st.sidebar.selectbox("Seleccione el genero", sexo_victima)
movil = st.sidebar.selectbox("Seleccione el movil de la victima", movil_victima)
dia_semana = st.sidebar.selectbox("Seleccione el dia de la semana", dias_semana)
horario = st.sidebar.selectbox("Seleccione el rango horario", rango_horario)
edad = st.sidebar.selectbox("Seleccione el rango de edad", rango_edad)

# Convertir entradas a los valores codificados
edad_cod = list(codEdad.transform([edad]))[0]
horario_cod = list(codHorario.transform([horario]))[0]
dia_cod = list(codDia.transform([dia_semana]))[0]
comuna_cod = list(codComuna.transform([comuna]))[0]
sexo_cod = list(codGenero.transform([sexo]))[0]

# Crear el vector de características a partir de los selectores
features = np.array([[sexo_cod, mes, comuna_cod, dia_cod, edad_cod, horario_cod]])

# Realizar la predicción
prediccion = modeloBA.predict(features)
probabilidad = modeloBA.predict_proba(features)

# Mostrar los resultados
st.subheader("Resultados de la Predicción")
st.write(f"Predicción: {prediccion[0]}")
st.write("Probabilidad de las clases:")
for i, clase in enumerate(modeloBA.classes_):
    st.write(f"- {clase}: {probabilidad[0][i]:.2%}")

# Mostrar análisis de delitos similares
dfc = df[(df["GENERO"] == sexo) & (df["MES_NUM"] == mes) & (df["NOM_COM"] == comuna) & (df["DIA_NOMBRE"] == dia_semana) & (df["TIPOLOGÍA"] == prediccion[0])]
solo = dfc['DELITO_SOLO'].value_counts() / dfc['DELITO_SOLO'].size
barrios = dfc['BARRIOS_HECHO'].value_counts() / dfc['BARRIOS_HECHO'].size

# Gráfico de pastel con las probabilidades
st.subheader("Gráfico de pastel de probabilidades")
fig, ax = plt.subplots()
ax.pie(probabilidad[0], labels=modeloBA.classes_, autopct='%1.1f%%', startangle=90)
ax.axis('equal')
st.pyplot(fig)

# Mostrar análisis de delitos similares cometidos
if len(solo) != 0:
    st.subheader("Análisis gráficos de delitos similares")
    st.write(dfc[['BARRIOS_HECHO', 'DESCRIPCION_CONDUCTA', 'ARMAS_MEDIOS', 'MOVIL_VICTIMA', 'DELITO_SOLO', 'MOVIL_AGRESOR', 'CLASE_SITIO']])
    
    st.write(f"Frecuencia en los barrios de la comuna {comuna}:")
    st.write(solo)
    
    st.write(f"Frecuencia en los barrios de la comuna {comuna}:")
    st.write(barrios)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(barrios, labels=barrios.index, autopct='%1.1f%%')
    st.pyplot(fig)