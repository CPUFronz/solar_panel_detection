import time
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor


class SVM:
    def __init__(self):
        estimators = 10
        self.classifier = BaggingRegressor(SVR(), max_samples=1.0/estimators, n_estimators=estimators, n_jobs=-1)
        self.regression_x = BaggingRegressor(SVR(), max_samples=1.0/estimators, n_estimators=estimators, n_jobs=-1)
        self.regression_y = BaggingRegressor(SVR(), max_samples=1.0/estimators, n_estimators=estimators, n_jobs=-1)
        self.regression_height = BaggingRegressor(SVR(), max_samples=1.0/estimators, n_estimators=estimators, n_jobs=-1)
        self.regression_width = BaggingRegressor(SVR(), max_samples=1.0/estimators, n_estimators=estimators, n_jobs=-1)

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        num_samples = X.shape[0]
        X = X.reshape(num_samples, -1)

        start = time.time()
        self.classifier.fit(X, y[:, 0])
        print('Training of Classifier done, took: {:5.2f}s'.format(time.time() - start))

        start = time.time()
        self.regression_x.fit(X, y[:, 1])
        print('Training of Regressor for X done, took: {:5.2f}s'.format(time.time() - start))

        start = time.time()
        self.regression_y.fit(X, y[:, 2])
        print('Training of Regressor for Y done, took: {:5.2f}s'.format(time.time() - start))

        start = time.time()
        self.regression_height.fit(X, y[:, 3])
        print('Training of Regressor for height done, took: {:5.2f}s'.format(time.time() - start))
        
        start = time.time()
        self.regression_width.fit(X, y[:, 4])
        print('Training of Regressor for width done, took: {:5.2f}s'.format(time.time() - start))

    def predict(self, X):
        l = len(X)
        X = X.reshape(l, -1)
        prediction = []

        start = time.time()
        prediction.append(self.classifier.predict(X))
        print('Prediction for confidence done, took: {:5.2f}s'.format(time.time() - start))

        start = time.time()
        prediction.append(self.regression_x.predict(X))
        print('Prediction for X done, took: {:5.2f}s'.format(time.time() - start))

        start = time.time()
        prediction.append(self.regression_y.predict(X))
        print('Prediction for Y done, took: {:5.2f}s'.format(time.time() - start))

        start = time.time()
        prediction.append(self.regression_height.predict(X))
        print('Prediction for height done, took: {:5.2f}s'.format(time.time() - start))

        start = time.time()
        prediction.append(self.regression_width.predict(X))
        print('Prediction for width done, took: {:5.2f}s'.format(time.time() - start))

        return np.array(prediction, dtype=np.float32).transpose()
