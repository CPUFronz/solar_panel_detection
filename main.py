import os
import time
import glob
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from constants import DATA_DIR
from constants import JSON_FILE
from constants import SAVED_MODEL
from constants import TEST_DATA
from utils import load_polygons
from utils import polygon2size
from utils import create_examples
from utils import conditioned_train_test_split
from svm import SVM

np.random.seed(42)

if __name__ == '__main__':
    start = time.time()

    if not os.path.exists(SAVED_MODEL):
        start_read_poly = time.time()
        polygons = load_polygons(JSON_FILE)
        print('Finished reading polygons, took{:5.2f}s'.format(time.time() - start_read_poly))

        start_read_files = time.time()
        tif_files = glob.glob(DATA_DIR + '*.tif')[:10]
        X, y = create_examples(tif_files, polygons)
        X, X_test, y, y_test = conditioned_train_test_split(X, y, test_size=0.2, min_positive_size=0.4)
        X = np.array(X)
        X_test = np.array(X_test)
        y = np.array(y)
        y_test = np.array(y_test)
        print('Finished creating {0:} training and {1:} test examples, took {2:5.2f}s'.format(len(X), len(X_test), time.time() - start_read_files))

        start_train = time.time()
        svm = SVM()
        svm.fit(X, y)

        with open(SAVED_MODEL, 'wb') as f:
            joblib.dump(svm, f)

        with open(TEST_DATA, 'wb') as f:
            joblib.dump((X_test, y_test), f)

        print('Finished training, took: {:5.2f}s'.format(time.time() - start_train))
    else:
        svm = joblib.load(SAVED_MODEL)
        X_test, y_test = joblib.load(TEST_DATA)
        print('Loaded trained model')

    predictions = svm.predict(X_test)

    error = mean_squared_error(y_test, np.array(predictions, dtype=np.float).squeeze())
    print('MSE:', error)
    print('Took:', time.time() - start)
