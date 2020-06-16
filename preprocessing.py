import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


class Preprocessor:
    def split(self, X, y, test_size):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        return X_train, X_test, y_train.astype(np.int64), y_test.astype(np.int64)

    def scaling(self, X_train, X_test):
        standardScaler = StandardScaler()

        standardScaler.fit(X_train)
        X_train = standardScaler.transform(X_train)
        X_test = standardScaler.transform(X_test)

        return X_train, X_test

    def dataCleaning(self, data):
        data.drop(data.columns[0], axis='columns', inplace=True)
        data_modified = data.drop(labels='price', axis=1)

        X = data_modified.iloc[:, 0:8].to_numpy()
        y = data.iloc[:, 6].values.reshape(53940, 1)

        return X, y

    def drop(self, data):
        data_modified = data.dropna(axis=0)

        return data_modified

    def encoding(self, X, y):
        labelEncoder = LabelEncoder()

        X[:, 1] = labelEncoder.fit_transform(X[:, 1])
        X[:, 2] = labelEncoder.fit_transform(X[:, 2])
        X[:, 3] = labelEncoder.fit_transform(X[:, 3])

        return X, y


