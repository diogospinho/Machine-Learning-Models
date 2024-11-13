import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Student_Performance.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

dataset.isnull().sum()

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

minhaprevisao = regressor.predict([[7,99,0,1,9,1]])
print(minhaprevisao)

