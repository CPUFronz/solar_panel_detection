import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from PIL import Image
from osgeo import gdal

from constants import IMG_SIZE_X
from constants import IMG_SIZE_Y
from constants import SVM_WINDOW_SIZE_X
from constants import SVM_WINDOW_SIZE_Y
from constants import DATA_DIR_USE
from constants import YOLO_IMG_SIZE_X
from constants import YOLO_IMG_SIZE_Y


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


def highlight_all_pvs(training_file):
    with open(training_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        splitted = line.strip().split(' ')
        filename = splitted.pop(0)

        img = Image.open(filename)
        img_array = np.array(img)
        plt.imshow(img_array)

        for s in splitted:
            coords = s.split(',')
            rect_x, rect_y = calc_rectangle(x_min=coords[0], y_min=coords[1], x_max=coords[2], y_max=coords[3])
            plt.plot(rect_x, rect_y, color='red')

        plt.show()


def highlight_pvs_center(filename, pvs):
    img = Image.open(filename)
    img_array = np.array(img)

    plt.imshow(img_array)
    for pv in pvs:
        rect_x, rect_y = calc_rectangle(x=pv.center_x, y=pv.center_y, h=pv.height, w=pv.width)
        plt.plot(rect_x, rect_y, color='red')

    plt.show()


def highlight_pvs_edges(filename, pvs):
    img = Image.open(filename)
    img_array = np.array(img)

#    plt.imshow(img_array)

    fig = plt.figure(figsize=[6, 6])
    ax = fig.add_subplot(111)
    ax.imshow(img_array)

    for pv in pvs:
        rect_x, rect_y = calc_rectangle(x_min=pv.x_min, y_min=pv.y_min, x_max=pv.x_max, y_max=pv.y_max)
#        plt.plot(rect_x, rect_y, color='red')
        ax.plot(rect_x, rect_y, color='red')

        if pv.score:
            plt.text(pv.x_min, pv.y_min - 2, '{:.2f}'.format(pv.score), color='red', fontsize=12)

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    plt.savefig(os.path.splitext(os.path.basename(filename))[0] + '.pdf', dpi=400, bbox_inches='tight', pad_inches=0)

#    plt.axis('off')
#    plt.savefig(os.path.splitext(os.path.basename(filename))[0] + '.png',  bbox_inches='tight', pad_inches=0)

    plt.show()


def calc_rectangle(**kwargs):
    if 'x_min' in kwargs and 'y_min' in kwargs and 'x_max' in kwargs and 'y_max' in kwargs:
        x_min = int(kwargs['x_min'])
        y_min = int(kwargs['y_min'])
        x_max = int(kwargs['x_max'])
        y_max = int(kwargs['y_max'])

        A = (x_min, y_min)
        B = (x_max, y_min)
        C = (x_max, y_max)
        D = (x_min, y_max)

    elif 'x' in kwargs and 'y' in kwargs and 'w' in kwargs and 'h' in kwargs:
        x = float(kwargs['x'])
        y = float(kwargs['y'])
        w = float(kwargs['w'])
        h = float(kwargs['h'])

        A = (int(x - w/2), int(y - h/2))
        B = (int(x + w/2), A[1])
        C = (B[0], int(y + h/2))
        D = (A[0], C[1])

    rect_x = [A[0], B[0], C[0], D[0], A[0]]
    rect_y = [A[1], B[1], C[1], D[1], A[1]]

    return rect_x, rect_y


def split_tif(filename, out_path=DATA_DIR_USE, step_x=YOLO_IMG_SIZE_X, step_y=YOLO_IMG_SIZE_Y, overwrite=False):
    src_ds = gdal.Open(filename)
    x_size = src_ds.RasterXSize
    y_size = src_ds.RasterYSize

    for x in range(0, x_size, step_x):
        for y in range(0, y_size, step_y):
            new_fn = out_path + os.path.splitext(os.path.basename(filename))[0] + '_{0:06d}_{1:06d}.tif'.format(x, y)

            if os.path.exists(new_fn) and not overwrite:
                continue
            else:
                print('Creating ' + os.path.basename(new_fn))
                new_ds = gdal.Translate(new_fn, src_ds, srcWin=[x, y, step_x, step_y])
                del new_ds


def create_training_data_window(filename, pvs, window_x=SVM_WINDOW_SIZE_X, window_y=SVM_WINDOW_SIZE_Y):
    img = Image.open(filename)
    img_array = np.array(img)

    X = []
    Y = []

    if IMG_SIZE_X != img_array.shape[0] or IMG_SIZE_Y != img_array.shape[1]:
        print(('Input image doesn\'t have the right size'), filename)
        return

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

            Y.append(target)

    return X, Y


def calc_intersection(A, B):
    xmin_A, ymin_A, xmax_A, ymax_A = A
    xmin_B, ymin_B, xmax_B, ymax_B = B

    dx = min(xmax_A, xmax_B) - max(xmin_A, xmin_B)
    dy = min(ymax_A, ymax_B) - max(ymin_A, ymin_B)

    intersection = dx * dy
    area_A = (xmax_A - xmin_A) * (ymax_A - ymin_A)
    area_B = (xmax_B - xmin_B) * (ymax_B - ymin_B)
    union = area_A + area_B - intersection

    return intersection / union


def calc_euclidean_distance(A, B):
    xmin_A, ymin_A, xmax_A, ymax_A = A
    xmin_B, ymin_B, xmax_B, ymax_B = B

    center_x_A = (xmax_A + xmin_A) / 2
    center_y_A = (ymax_A + ymin_A) / 2
    center_x_B = (xmax_B + xmin_B) / 2
    center_y_B = (ymax_B + ymin_B) / 2

    return np.sqrt((center_x_A - center_x_B)**2 + (center_y_A - center_y_B)**2)


def calc_true_positives(y_true, y_predicted):
    def parse_values(values):
        parsed = {}
        for val in values:
            split = val.strip().split(' ')
            parsed[split[0]] = [[int(b) for b in box] for boxes in split[1:] for box in [boxes.strip(',0').split(',')]]
        return parsed

    y_true = parse_values(y_true)
    y_predicted = parse_values(y_predicted)

    true_positives = 0
    total_number_objects = 0
    for key in y_true:
        yt = y_true[key]
        yp = y_predicted[key]

        total_number_objects += len(yt)

        for i, _ in enumerate(yt):
            for j, _ in enumerate(yp):
                if calc_euclidean_distance(yt[i], yp[j]) < 25 and 1 > calc_intersection(yt[i], yp[j]) > 0.25:
                    true_positives += 1

    return true_positives, total_number_objects

# TODO: implement https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
#                 https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173


if __name__ == '__main__':
    split_tif('/home/franz/workspace/solar_panel_detection/data/use/Vienna_2017.tif')
