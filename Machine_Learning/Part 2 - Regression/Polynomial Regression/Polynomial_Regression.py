# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:19:31 2020

@author: Nik Nikhil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree = 4)
X_poly = poly_features.fit_transform(X)
poly_features.fit(X_poly, Y)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_poly, Y)

plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(poly_features.fit_transform(X_grid)), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('Levels')
plt.ylabel('Salary')
plt.show()

regressor.predict(poly_features.fit_transform([[10]]))
