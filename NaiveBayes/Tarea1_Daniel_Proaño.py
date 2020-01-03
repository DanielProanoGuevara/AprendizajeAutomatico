# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 12:36:37 2019
Naive Bayes
@author: Daniel Proaño
"""

## Preprocesamiento de Datos
#Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar la base de datos
dataset  = pd.read_csv('house-votes-84.data')
#Separa la última columna que es la data de resultados
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

#Tratamiento de missing
#Reemplaza los valores de ? por el valor más frecuente en la columna
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = '?', strategy = 'most_frequent')
imputer = imputer.fit(X) 
X = imputer.transform(X)

#Codificación de datos categóricos
from sklearn.preprocessing import LabelEncoder
#Crea el modelo de codificación
labelencoder_X = LabelEncoder()
#Aplica el modelo de codificación en todos los datos
X[:, 15] = labelencoder_X.fit_transform(X[:, 15])
for i in range(15):
    X[:, i] = labelencoder_X.transform(X[:, i])
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Separar los conjuntos de datos en grupo de entrenamiento y de prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# Ajustar el clasificador con los datos de entrenamiento
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predecir los resultados en base a los datos de prueba
y_pred = classifier.predict(X_test)

# Crea la matriz de confusión comparando los datos predichos y los datos reales
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Crea un dataframe con los datos predichos y reales
y_test_pd = pd.Series(y_test, name = 'Real')
y_pred_pd = pd.Series(y_pred, name = 'Predicho')
df_confusion = pd.crosstab(y_test_pd, y_pred_pd)
print("Matriz de confusión")
print(df_confusion)
print("1 --> Republicano, 0 --> Demócrata")
print('\n')
#print(cm)


#Imprime métricas de evaluación del modelo
from sklearn import metrics
print('Precisión del modelo')
print(metrics.accuracy_score(y_test, y_pred))
print('Reporte de clasificación')
print(metrics.classification_report(y_test, y_pred))
print('\n')

#Construcción de ROC
#Datos para curva de referencia
ns_probs = [0 for _ in range(len(y_test))]
#Predecir probabilidades
lr_probs = classifier.predict_proba(X_test)
#Mantiene solo las probabilidades positivas del modelo
lr_probs = lr_probs[:, 1]
# calcula las métricas del modelo
ns_auc = metrics.roc_auc_score(y_test, ns_probs) #Curva sin entrenamiento
lr_auc = metrics.roc_auc_score(y_test, lr_probs) #Curva entrenada
#Resume puntuaciones de área bajo la curva
print('Sin entrenamiento: ROC AUC=%.3f' % (ns_auc))
print('Modelo entrenado: ROC AUC=%.3f' % (lr_auc))
#Calcula curvas ROC
ns_fpr, ns_tpr, _ = metrics.roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = metrics.roc_curve(y_test, lr_probs)
#Grafica la curva ROC para el modelo
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Sin entrenamiento')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Modelo entrenado')
# axis labels
plt.xlabel('Razón de falsos positivos')
plt.ylabel('Razón de falsos negativos')
# show the legend
plt.legend()
# show the plot
plt.show()