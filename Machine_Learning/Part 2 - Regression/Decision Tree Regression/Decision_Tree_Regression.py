# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:50:20 2020

@author: Nik Nikhil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X, Y)

regressor.predict([[6.5]])

X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape(len(X_grid), 1)


plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()