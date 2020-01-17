# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:33:23 2020
Decision Tree Regression
@author: dproaño
"""

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv('hour.csv')
X = dataset.iloc[:, 0:16].values
y = dataset.iloc[:, 16].values

#Tratamiento de data
from datetime import datetime
for i in range(len(X[:,1])):
   X[i,1] = datetime.fromisoformat(X[i,1]).timestamp() 

#Split dataset into Training & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))
y_test = sc_y.transform(y_test.reshape(-1, 1))

#Fitting the Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

#Predicting a new result
y_pred = regressor.predict(X_test)

#Regression results
MAE=0
MSE=0
RMSE=0
num_arb=25
aux=25
from sklearn import metrics
MAE=np.append(MAE,metrics.mean_absolute_error(y_test, y_pred))
MSE=np.append(MSE,metrics.mean_squared_error(y_test, y_pred))
RMSE=np.append(RMSE,np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
aux=aux+25
num_arb=np.append(num_arb,aux)