from sklearn import metrics
from sklearn.linear_model import LinearRegression


class Regression:
    def __init__(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        linear_regressor = LinearRegression()

        linear_regressor.fit(self.X_train, self.y_train)

        return linear_regressor.predict(X_test)

    def getScore(self, y_test, y_pred):
        return metrics.r2_score(y_test, y_pred)

