import os
import time
import glob
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from constants import DATA_DIR_USE
from yolowrapper import YoloWrapper

np.random.seed(42)

if __name__ == '__main__':
    yolo = YoloWrapper()
    yolo.train()

    for f in glob(DATA_DIR_USE + '*.tif'):
        yolo.predict(f)
    yolo.close_session()