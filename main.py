import os
import time
import glob
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from constants import DATA_DIR_USE
from constants import YOLO_MODEL
from constants import YOLO_TRAINING_DATA
from yolowrapper import YoloWrapper
from utils import calc_true_positives

np.random.seed(42)

if __name__ == '__main__':
    yolo = YoloWrapper()
    if not os.path.exists(YOLO_MODEL):
        yolo.train()

    with open(YOLO_TRAINING_DATA, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    validation_examples = lines[:10] #np.random.choice(lines, 10)

    predictions = []
    validation_files = [v.split(' ')[0] for v in validation_examples]
    for f in validation_files:
        predictions.append(yolo.predict(f))
    yolo.close_session()

    tmp = []
    for p in predictions:
        line = p[0] + ' '
        for boxes in p[1]:
            for b in boxes:
                line += str(int(b)) + ','
            line += '0 '

        tmp.append(line)
    predictions = tmp

    print(calc_true_positives(validation_examples, predictions))