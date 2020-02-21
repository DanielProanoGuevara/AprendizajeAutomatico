# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:18:55 2020

@author: Daniel Proaño
"""

#Importar Librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el dataset con pandas
dataset = pd.read_csv('Wholesale_customers_data.csv')
# Se excluyen las variables categóricas de Channel y Region
X = dataset.iloc[:,2:].values

#Describe el dataset 
print(dataset.iloc[:,2:].describe())

#Se utiliza el método de codo para determinar la mejor cantidad de clusters
from sklearn.cluster import KMeans
sse = []
for i in range (1,21):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)
plt.figure()
plt.plot(range(1, 21), sse)
plt.title('Método de codo')
plt.xlabel('Número de clusters')
plt.ylabel('SSE')
plt.grid(True)
plt.show()


#Aplicar clustering de k-means sobre el dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)


#Visualizar los clusters
plt.figure()
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'cluster 3')

#Centroides
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'centroides')
plt.title('Clusters de clientes')
plt.xlabel('Fresh')
plt.ylabel('Milk')
plt.legend()
plt.show()