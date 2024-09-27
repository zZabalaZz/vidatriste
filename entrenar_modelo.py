#python -c "import sklearn; sklearn.show_versions()
#pip uninstall scikit-learn --yes
#pip install --upgrade scikit-learn
#Verificar la version de entrenaiento
  #1 #TRATAMIENTO DE DATOS
import pandas as pd
import numpy as np

#SISTEMA OPERATIVO
import os

#GRAFICO
import matplotlib.pyplot as plt


#MAPA DE CALOR


from sklearn import preprocessing
import joblib as jb

from sklearn.metrics import confusion_matrix

from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier


dfn=pd.read_csv("C:/Users/adiaz/Documents/IA/Delitos/delitos_minable.csv")


codEdad=preprocessing.LabelEncoder()
dfn['RangoEdad']=codEdad.fit_transform(dfn['RangoEdad'])
print(codEdad.classes_)
jb.dump(codEdad,'C:/Users/adiaz/Documents/IA/Delitos/codEdad.bin')


codHorario=preprocessing.LabelEncoder()
dfn['rangoHORARIO']=codHorario.fit_transform(dfn['rangoHORARIO'])
print(codHorario.classes_)
jb.dump(codHorario,'C:/Users/adiaz/Documents/IA/Delitos/codHorario.bin')


codComuna=preprocessing.LabelEncoder()
dfn['NOM_COM']=codComuna.fit_transform(dfn['NOM_COM'])
print(codComuna.classes_)
jb.dump(codComuna,'C:/Users/adiaz/Documents/IA/Delitos/codComuna.bin')

codDia=preprocessing.LabelEncoder()
dfn['DIA_NOMBRE']=codDia.fit_transform(dfn['DIA_NOMBRE'])
print(codDia.classes_)
jb.dump(codDia,'C:/Users/adiaz/Documents/IA/Delitos/codDia.bin')


codGenero=preprocessing.LabelEncoder()
dfn['GENERO']=codGenero.fit_transform(dfn['GENERO'])
print(codGenero.classes_)
jb.dump(codGenero,'C:/Users/adiaz/Documents/IA/Delitos/codGenero.bin')


from sklearn.model_selection import train_test_split
X=dfn.drop(['TIPOLOGIA2','Unnamed: 0'],axis=1)#features
y=dfn['TIPOLOGIA2']#label
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=1234,shuffle=True)

modeloBA= RandomForestClassifier(random_state=0,
                                 criterion='entropy',
                                 n_estimators=100,
                                 max_features="sqrt",
                                 bootstrap=True,
                                 max_samples=3/4,
                                 oob_score=True)
#Entreno el modelo

modeloBA.fit(X_train, y_train)

print("Score de predicción", modeloBA.fit(X_train, y_train).predict_proba(X_test))
print("Accuracy para los datos de entrenamiento {0:.2f}%".format(modeloBA.score(X_train,y_train)*100))
print("Accuracy para los datos de oob {0:.2f}%".format(modeloBA.oob_score_*100))
print("Accuracy para los datos de prueba {0:.2f}%".format(modeloBA.score(X_test,y_test)*100))


#Predicción con los datos de prueba
y_predict=modeloBA.predict(X_test)
print("Reporte del Score del modelo",classification_report(y_test,y_predict))

#Comparar los resultado
y_predict_df=pd.DataFrame(y_predict,columns=["prediccion"])
y_test_oredena= y_test.reset_index(drop=True)
comparativo=pd.concat([y_test_oredena,y_predict_df], axis=1)
print(comparativo.head(20))
print("Score de predicción", modeloBA.fit(X_train, y_train).predict_proba(X_test))


matrix=confusion_matrix(y_test,y_predict,labels=modeloBA.classes_)
displaymatrix=ConfusionMatrixDisplay(confusion_matrix=matrix,display_labels=modeloBA.classes_)
displaymatrix.plot(xticks_rotation='vertical')
plt.show()

jb.dump(modeloBA,"C:/Users/adiaz/Documents/IA/Delitos/modeloBA.bin")
print(modeloBA.feature_names_in_)