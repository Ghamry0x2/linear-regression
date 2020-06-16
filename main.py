import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from regression import Regression
from preprocessing import Preprocessor

# Data Reading
dataSet = pd.read_csv('diamonds.csv')

# Data Preprocessing
preprocessing = Preprocessor()

X_regression, y_regression = preprocessing.dataCleaning(dataSet)
X_regression, y_regression = preprocessing.encoding(X_regression, y_regression)
dataSet = preprocessing.drop(dataSet)

X_train, X_test, y_train, y_test = preprocessing.split(X_regression, y_regression, 0.3)
X_train, X_test = preprocessing.scaling(X_train, X_test)

# Linear Regression
regression = Regression(X_train, np.ravel(y_train))

y_predicted = regression.predict(X_test)
score = regression.getScore(np.ravel(y_test), y_predicted)

# Outputs
pd.set_option('display.max_columns', 500)
print(dataSet.describe())
print('\nThe score is: ' + str(score*100) + "%")

# plt.boxplot(dataSet['x'], vert=False)
# plt.show()
# pd.plotting.scatter_matrix(dataSet)
# plt.scatter(dataSet['x'], dataSet['price'])
# plt.show()
