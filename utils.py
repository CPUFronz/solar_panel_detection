import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from PIL import Image

from constants import IMG_SIZE_X
from constants import IMG_SIZE_Y
from constants import WINDOW_SIZE_X
from constants import WINDOW_SIZE_Y


def load_polygons(json_file):
    with open(json_file) as f:
        polygons = json.load(f)

    polygons = [p for p in polygons['polygons']]

    solar_pv = namedtuple('solar_pv', 'id center_x center_y height width filename')
    findings = []

    for p in polygons:
        polygon_id = p['polygon_id']
        center_x = p['centroid_longitude_pixels']
        center_y = p['centroid_latitude_pixels']
        point_list = p['polygon_vertices_pixels']
        width, height = polygon2size(point_list)
        filename = p['image_name']
        pv = solar_pv(polygon_id, center_x, center_y, height, width, filename)
        findings.append(pv)

    return findings


def create_examples(tif_files, polygons):
    cnt = 0
    X = []
    y = []

    for tif in tif_files:
        pvs = []
        for pv in polygons:
            # only create a list of pvs for the current file, saves a lot of time
            if os.path.basename(tif).replace('.tif', '') == pv.filename:
                # append only if center is not _NaN_
                if type(pv.center_x) != str or type(pv.center_y) != str:
                    pvs.append(pv)

        train_x, train_y = create_training_data_image(tif, pvs)
        X.extend(train_x)
        y.extend(train_y)
        cnt += 1
        print('Created trainig data: {0:3d}/{1:3d}'.format(cnt, len(tif_files)))

    return X, y


def polygon2size(points):
    x_min = IMG_SIZE_X
    y_min = IMG_SIZE_Y
    x_max = 0
    y_max = 0

    for point in points:
        if type(point) == float:
            return 1, 1

        if point[0] < x_min:
            x_min = point[0]
        if point[1] < y_min:
            y_min = point[1]
        if point[0] > x_max:
            x_max = point[0]
        if point[1] > y_max:
            y_max = point[1]

    return x_max - x_min, y_max - y_min


def highlight_pv(filename, pvs):
    img = Image.open(filename)
    img_array = np.array(img)

    plt.imshow(img_array)
    for pv in pvs:
        x = pv.center_x
        y = pv.center_y
        h = pv.height
        w = pv.width

        A = (int(x - w/2), int(y - h/2))
        B = (int(x + w/2), A[1])
        C = (B[0], int(y + h/2))
        D = (A[0], C[1])

        rect_x = [A[0], B[0], C[0], D[0], A[0]]
        rect_y = [A[1], B[1], C[1], D[1], A[1]]

        plt.plot(rect_x, rect_y, color='red')

    plt.show()


def create_training_data_image(filename, pvs, window_x=WINDOW_SIZE_X, window_y=WINDOW_SIZE_Y):
    img = Image.open(filename)
    img_array = np.array(img)

    X = []
    Y = []

    if IMG_SIZE_X != img_array.shape[0] or IMG_SIZE_Y != img_array.shape[1]:
        print(('Input image doesn\'t have the right size'), filename)
        return

    for x in range(0, IMG_SIZE_X, WINDOW_SIZE_X):
        for y in range(0, IMG_SIZE_Y, WINDOW_SIZE_Y):
            x_end = x + WINDOW_SIZE_X
            y_end = y + WINDOW_SIZE_Y
            X.append(img_array[x:x_end, y:y_end, :3])

            # target = [confidence, x, y, h, w]
            target = np.array([0, 0, 0, 0, 0], dtype=np.float32)

            for pv in pvs:
                if x <= pv.center_x <= x_end and y <= pv.center_y <= y_end:
                    x_relative = (pv.center_x - x) / WINDOW_SIZE_X
                    y_relative = (pv.center_y - y) / WINDOW_SIZE_Y
                    target = np.array([1, x_relative, y_relative, pv.height, pv.width], dtype=np.float32)

            Y.append(target)

    return X, Y


def conditioned_train_test_split(X, y, test_size, min_positive_size):
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

    return X_train, X_test, y_train, y_test
