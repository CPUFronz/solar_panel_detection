import time
import numpy as np
from glob import glob
from PIL import Image
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor

import utils
from constants import IMG_SIZE_X
from constants import IMG_SIZE_Y
from constants import SVM_WINDOW_SIZE_X
from constants import SVM_WINDOW_SIZE_Y
from constants import DATA_DIR_TRAIN
from constants import JSON_FILE

class SVM:
    def __init__(self):
        estimators = 10
        self.classifier = BaggingRegressor(SVR(), max_samples=1.0/estimators, n_estimators=estimators, n_jobs=-1)
        self.regression_x = BaggingRegressor(SVR(), max_samples=1.0/estimators, n_estimators=estimators, n_jobs=-1)
        self.regression_y = BaggingRegressor(SVR(), max_samples=1.0/estimators, n_estimators=estimators, n_jobs=-1)
        self.regression_height = BaggingRegressor(SVR(), max_samples=1.0/estimators, n_estimators=estimators, n_jobs=-1)
        self.regression_width = BaggingRegressor(SVR(), max_samples=1.0/estimators, n_estimators=estimators, n_jobs=-1)

    def load_training_data(self, image_dir=DATA_DIR_TRAIN, polygon_file=JSON_FILE):
        tif_files = glob(image_dir + '*.tif')
        polygons = utils.load_polygons(polygon_file)

        X = []
        y = []
        for tif in tif_files:
            data = self.create_training_data_image(tif, polygons)
            X.extend(data[0])
            y.extend(data[1])

        self.conditioned_train_test_split(X, y, 0.2, 0.4)

    def create_training_data_image(self, filename, pvs, window_x=SVM_WINDOW_SIZE_X, window_y=SVM_WINDOW_SIZE_Y):
        # TODO: add meaningful features

        img = Image.open(filename)
        img_array = np.array(img)

        if IMG_SIZE_X != img_array.shape[0] or IMG_SIZE_Y != img_array.shape[1]:
            print(('Input image doesn\'t have the right size'), filename)
            return

        X = []
        y = []

        for x in range(0, IMG_SIZE_X, SVM_WINDOW_SIZE_X):
            for y in range(0, IMG_SIZE_Y, SVM_WINDOW_SIZE_Y):
                x_end = x + SVM_WINDOW_SIZE_X
                y_end = y + SVM_WINDOW_SIZE_Y
                X.append(img_array[x:x_end, y:y_end, :3])

                # target = [confidence, x, y, h, w]
                target = np.array([0, 0, 0, 0, 0], dtype=np.float32)

                for pv in pvs:
                    if x <= pv.center_x <= x_end and y <= pv.center_y <= y_end:
                        x_relative = (pv.center_x - x) / SVM_WINDOW_SIZE_X
                        y_relative = (pv.center_y - y) / SVM_WINDOW_SIZE_Y
                        target = np.array([1, x_relative, y_relative, pv.height, pv.width], dtype=np.float32)

                y.append(target)

        return X, y

    def conditioned_train_test_split(self, X, y, test_size, min_positive_size):
        num_total = len(X)
        num_test = int(num_total * test_size)
        num_test_positive = int(num_test * min_positive_size)

        perm = np.random.permutation(num_total)
        X = [X[i] for i in perm]
        y = [y[i] for i in perm]

        X_train = []
        y_train = []
        X_test = []
        y_test = []

        indices = []
        for i in range(len(X)):
            if y[i][0] == 1:
                X_test.append(X[i])
                y_test.append(y[i])
                indices.append((i))
            if len(y_test) >= num_test_positive:
                break

        num_test_positive_actual = len(y_test)
        if num_test_positive_actual <= num_test_positive:
            num_train_positive = int((1 - min_positive_size) * num_test_positive_actual)
            for i in reversed(range(num_train_positive)):
                X_train.append(X_test.pop(i))
                y_train.append(y_test.pop(i))
                del indices[i]

        for idx in reversed(indices):
            del X[idx]
            del y[idx]

        num_test = int(len(y_test) / min_positive_size)
        num_train = int(num_test / test_size)

        while len(X_test) <= num_test:
            idx = np.random.randint(len(X))
            X_test.append(X[idx])
            y_test.append(y[idx])

            del X[idx]
            del y[idx]

        # pruning negative training examples
        while len(X_train) <= num_train:
            idx = np.random.randint(len(X))
            X_train.append(X[idx])
            y_train.append(y[idx])

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)

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
