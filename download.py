import os
import urllib.request
from zipfile import ZipFile
from multiprocessing.pool import ThreadPool
from osgeo import gdal

import utils
from constants import YOLO_IMG_SIZE_X
from constants import YOLO_IMG_SIZE_Y
from constants import DATA_DIR_USE
from constants import WMTS_LAYERS
from constants import WMTS_XML
from constants import TRAINING_DATA_URLS
from constants import DATA_DIR_TRAIN
from constants import JSON_FILE


def download_layer(layer, window_x=1, window_y=1):
    def download_tile(inner_args):
        layer, x, y = inner_args
        ds = gdal.Open(WMTS_XML.format(layer))
        new_ds = gdal.Translate(DATA_DIR_USE + layer + '/{0:}_{1:05d}_{2:05d}.tif'.format(layer, x, y), ds,
                                srcWin=[x * window_x, y * window_y, window_x, window_y])
        del new_ds

    if not os.path.exists(DATA_DIR_USE + layer):
        os.mkdir(DATA_DIR_USE + layer)

    src_ds = gdal.Open(WMTS_XML.format(layer))
    x_max = src_ds.RasterXSize // window_x
    y_max = src_ds.RasterYSize // window_y
    del src_ds

    args = []
    for x in range(x_max):
        for y in range(y_max):
            args.append((layer, x, y))

    p = ThreadPool()
    p.map(download_tile, args)
    p.close()
    p.join()


def download_wmts_data(window_x=1, window_y=1):
    layers = [WMTS_LAYERS[-1], WMTS_LAYERS[0]]
    for layer in layers[0] + layers[-1]:
        print('Downloading', layer)
        download_layer(layer, YOLO_IMG_SIZE_X, YOLO_IMG_SIZE_Y)
        print('Finished downloading', layer)


def download_training_data(remove_source=True):
    def show_progress(count, block_size, total_size):
        percent = ((count * block_size) / total_size) * 100
        print('Downloading {0:}: {1:.1f}%'.format(url, percent), end='\r')

    for fn, url in TRAINING_DATA_URLS.items():
        path = DATA_DIR_TRAIN + '../' + fn
        urllib.request.urlretrieve(url, path, show_progress)

        with ZipFile(path) as zipf:
            if fn == 'polygons.zip':
                zipf.extract(os.path.basename(JSON_FILE), path=os.path.dirname(path))
            else:
                zipf.extractall(path=DATA_DIR_TRAIN)

        if remove_source:
            os.remove(path)

        print()


def download_data():
    download_training_data()
    download_wmts_data(YOLO_IMG_SIZE_X, YOLO_IMG_SIZE_Y)


if __name__ == '__main__':
    download_data()
