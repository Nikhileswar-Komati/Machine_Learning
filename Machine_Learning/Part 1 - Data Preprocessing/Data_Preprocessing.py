import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 0:3].values
Y = dataset.iloc[:, 3].values


from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
columntransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder = "passthrough")
X = np.array(columntransformer.fit_transform(X), dtype = np.float)

labelencoder = LabelEncoder()

Y = labelencoder.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)