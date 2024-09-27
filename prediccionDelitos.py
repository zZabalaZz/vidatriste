import streamlit as st

#importar las bibliotecas tradicionales de numpy y pandas
import numpy as np
import pandas as pd

#importar las biliotecas graficas e imágenes
import plotly.express as px
from PIL import Image
import matplotlib.pyplot as plt


import joblib as jb

from sklearn.ensemble import RandomForestClassifier



imagen_video = Image.open("delitos-federales.jpg") 


#Librerias no usadas
#from streamlit_lottie import st_lottie
#import requests

## Iniciar barra lateral en la página web y título e ícono

st.set_page_config(
  page_title="ML Delitos Bucaramanga",
  page_icon="4321369.png",
  initial_sidebar_state='auto'
  )

@st.cache_data
def load_data():
  df= pd.read_csv('Delito Bucaramanga_preprocesar.csv', delimiter=",") #Currently on my local machine
  return df
df= load_data()

@st.cache_resource
def load_models():
  codEdad=jb.load('codEdad.bin')
  codHorario=jb.load('codHorario.bin')
  codGenero=jb.load('codGenero.bin')
  codDia=jb.load('codDia.bin')
  codComuna=jb.load('codComuna.bin')
  modeloBA=jb.load('modeloBA.bin')
  return codEdad,codHorario,codGenero,codDia,codComuna,modeloBA
codEdad,codHorario,codGenero,codDia,codComuna,modeloBA = load_models()

#Primer contenedor
with st.container():
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
st.markdown("""
### Variables de Entrada y su Utilidad en la Predicción:
- *Comuna:* Bucaramanga está dividida en 17 comunas, cada una con características sociodemográficas y geográficas particulares.
- *Mes:* El mes en el que ocurre el delito permite identificar patrones estacionales.
- *Sexo de la víctima:* El género de la víctima puede influir en la probabilidad de ciertos tipos de delitos.
- *Móvil de la víctima:* Este factor describe la situación o actividad en la que se encontraba la víctima al momento del delito.
- *Día de la semana:* Algunos delitos pueden ocurrir con mayor frecuencia en días específicos.
- *Rango de horario:* El momento del día* en que ocurre un delito es fundamental, ya que ciertos delitos suelen suceder en horas específicas.
- *Curso de vida de la víctima:* La edad de la víctima puede ser un factor clave para entender la vulnerabilidad a ciertos tipos de crímenes.
""")
st.markdown("---")
#Librerias usadas
st.markdown("""
### Librerias Usadas para el modelo:
- *numpy*
- *pandas*
- *plotly*
- *requests*
- *streamlit*
- *matplotlib*
- *scikit-learn*
""")

st.markdown("---")
st.markdown("""
El conjunto de datos cargado corresponde a los *registros de delitos en Bucaramanga*. A continuación, se presenta un resumen de las principales características de los datos:
""")

st.write("El número total de registros cargados es: ", len(df))

# Mostrar las primeras filas del dataset para vista previa
st.subheader("Vista Previa de los Datos")
st.dataframe(df.head(5))  # Mostrar las primeras 5 filas del dataframe

      
edades=['ADOLECENCIA','ADULTEZ','INFANCIA','JUVENTUD','PERSONA MAYOR','PRIMERA INFANCIA']
horas=['MADRUGADA','MAÑANA','NOCHE','TARDE']
comunas=['CABECERA DEL LLANO','CENTRO', 'GARCIA ROVIRA', 'LA CIUDADELA',
 'LA CONCORDIA', 'LA PEDREGOSA', 'LAGOS DEL CACIQUE', 'MORRORICO', 'MUTIS',
 'NORORIENTAL', 'NORTE', 'OCCIDENTAL', 'ORIENTAL', 'PROVENZA', 'SAN FRANCISCO',
 'SUR', 'SUROCCIDENTE']
generos=['FEMENINO','MASCULINO']
diaSemana=['lunes','martes','miércoles','jueves','sábado','viernes','domingo']

st.subheader("Detalle del dataset usado en el proyecto")

st.write("El número de registros cargados es: ", len(df))
#st.write("comprendido desde ", pd.to_datetime(df['FECHA_COMPLETA']).min(), " hasta ", pd.to_datetime(df['FECHA_COMPLETA']).max())
st.write("El númerp de tipos de delitos registrados  ", len(df['DELITO_SOLO'].unique()), ", de", len(df['BARRIOS_HECHO'].unique()), "barrios en ",len(df['NOM_COM'].unique()),  " comunas")
st.write(df.head(5))
#Opciones de la barra lateral

logo=Image.open("menu.jpg")
st.sidebar.image(logo, width=100)
st.sidebar.header('Seleccione los datos de entrada')


def seleccionar(generos,comunas, diaSemana,edades,horas):

  #Filtrar por municipio

  st.sidebar.subheader('Selector del Género')
  genero=st.sidebar.selectbox("Seleccione el genero",generos)

  #Filtrar por estaciones
  st.sidebar.subheader('Selector del dia de la semana')
  dia=st.sidebar.selectbox("Selecciones del dia de la semana",diaSemana)
  
  #Filtrar por estaciones
  st.sidebar.subheader('Selector del dia del rango de edad')
  edad=st.sidebar.selectbox("Selecciones la edad",edades)
  
  #Filtrar por estaciones
  st.sidebar.subheader('Selector del rengo de dia')
  hora=st.sidebar.selectbox("Seleccione la jornada ",horas)
  
  st.sidebar.subheader('Selector de mes') 
  mes=st.sidebar.slider('número del mes', 1, 12, 1)
  
  #Filtrar por departamento
  st.sidebar.subheader('Selector de comuna')
  comuna=st.sidebar.selectbox("Seleccione la comuna",comunas)

  
  return edad,genero,mes,hora,comuna,dia

edad,genero,mes,hora,comuna,dia=seleccionar(generos,comunas,diaSemana,edades,horas)



#st.write(datos.describe())
with st.container():
  st.subheader("Predición")
  st.title("Predicción de Articulo del Código Civil Colombiano")
  st.write("""
           El siguiente es el pronóstico de la clase delito usando el modelo usando los diferentes umbrales
           """)
           
  edadn=list(codEdad.transform([edad]))[0]
  horan=list(codHorario.transform([hora]))[0]
  dian=list(codDia.transform([dia]))[0]
  comunan=list(codComuna.transform([comuna]))[0]
  generon=list(codGenero.transform([genero]))[0]
  lista=[[generon,mes,comunan,dian,edadn,horan]]
  
  st.write("Se han seleccionado los siguientes parámetros:")
  st.write("Edad: ", edad, "equvalente a",edadn )
  st.write("Género : ", genero,"equvalente a",generon)
  st.write("Mes :", mes,"equvalente a",mes)
  st.write("Hora", hora,"equvalente a", horan)
  st.write("Comuna",comuna,"equvalente a", comunan)
  st.write("dia",dia,"equvalente a",dian) 
  
  X_predecir=pd.DataFrame(lista,columns=['GENERO','MES_NUM','NOM_COM','DIA_NOMBRE','RangoEdad','rangoHORARIO'])
  y_predict=modeloBA.predict(X_predecir)
  st.markdown('----')
  st.markdown("<h1 style='color: #340467;'>La predicción es:</h1>", unsafe_allow_html=True)
  st.title(y_predict[0])
  st.markdown('----')
  dfc=df[(df["GENERO"]==genero) & (df["MES_NUM"]==mes) & (df["NOM_COM"]==comuna) & (df["DIA_NOMBRE"]==dia) & (df["TIPOLOGÍA"]==y_predict[0])]
  solo=dfc['DELITO_SOLO'].value_counts()/dfc['DELITO_SOLO'].size
  solo.rename({'count':'Frecuencia'}, inplace = True)
  barrios=dfc['BARRIOS_HECHO'].value_counts()/dfc['BARRIOS_HECHO'].size
  barrios.rename({'count':'Frecuencia'}, inplace = True)
  
with st.container():
  if len(solo)!=0:
    st.subheader("Análisis gráficos")
    st.subheader("Delitos similares cometidos con los parámetros dados")
    st.write("""
           Para apoyar el análisis y la toma de decisiones se presentan los delites cometidos en esas
           opciones.
           """)

    st.write(dfc[['BARRIOS_HECHO','DESCRIPCION_CONDUCTA', 'ARMAS_MEDIOS',
        'MOVIL_VICTIMA','DELITO_SOLO', 'MOVIL_AGRESOR', 'CLASE_SITIO']])
    
    st.write('Frecuencia en los barrios de la comuna  '+ comuna + '  es: ', solo )
    
    st.write('Frecuencia en los barrios de la comuna  '+ comuna + '  es :',barrios )
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(barrios,labels=barrios.index, autopct='%1.1f%%')
    st.pyplot(fig)
