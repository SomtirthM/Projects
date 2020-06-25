# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as sklm

# Importing the dataset
rd = pd.read_csv('DataNew.csv')
X = rd.iloc[:, :-1].values
y = rd.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 14)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
# Train with all the records

lm = sklm.Ridge(alpha = 0.014, normalize = True)
lm.fit(X_train, y_train)
# Prediction of train data
pred_train = lm.predict(X_train)
# Prediction of test data
pred_test = lm.predict(X_test)


# Fitting Simple Linear Regression to the Training set


axes = fig.add_subplot(1, 2, 1)
axes.plot(pred_train, label = 'Train data - predicted value')
axes.plot(y_train, label = 'Train data - actual value')
pp.xlabel('Player')
pp.ylabel('Transfer fee')
pp.title('Prediction on train data')
pp.legend()
#
axes = fig.add_subplot(1, 2, 2)
axes.plot(pred_test, label = 'Test data - predicted value')
axes.plot(y_test, label = 'Test data - actual value')
pp.xlabel('Player')
pp.ylabel('Transfer fee')
pp.title('Prediction on test data')
pp.legend()
