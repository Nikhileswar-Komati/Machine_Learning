# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 12:51:49 2020

@author: Nik Nikhil
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_y = StandardScaler()
X = scaler_x.fit_transform(X)
Y = scaler_y.fit_transform(Y.reshape(-1, 1)) 

from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X, Y)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

y_pred = scaler_y.inverse_transform(regressor.predict(scaler_x.transform(np.array([[6.5]]))))

plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()