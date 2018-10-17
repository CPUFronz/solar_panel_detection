import os
import urllib.request
from zipfile import ZipFile
from multiprocessing.pool import ThreadPool
from osgeo import gdal

from constants import YOLO_IMG_SIZE_X
from constants import YOLO_IMG_SIZE_Y
from constants import DATA_DIR_USE
from constants import WMTS_LAYERS
from constants import WMTS_XML
from constants import TRAINING_DATA_URLS
from constants import DATA_DIR_TRAIN
from constants import JSON_FILE


def download_layer(layer, out_path, window_x, window_y, overwrite=False):
    def download_tile(inner_args):
        layer, fn, x, y = inner_args
        ds = gdal.Open(WMTS_XML.format(layer))
        new_ds = gdal.Translate(fn, ds, srcWin=[x * window_x, y * window_y, window_x, window_y])
        del new_ds
        print('Downloaded', fn)

    src_ds = gdal.Open(WMTS_XML.format(layer))
    x_max = src_ds.RasterXSize // window_x
    y_max = src_ds.RasterYSize // window_y
    del src_ds

    args = []
    for x in range(x_max):
        for y in range(y_max):
            fn = DATA_DIR_USE + '{0:}_{1:05d}_{2:05d}.tif'.format(layer, x, y)
            if os.path.exists(fn) and not overwrite:
                continue

            args.append((layer, fn, x, y))

    p = ThreadPool(16)
    p.map(download_tile, args)
    p.close()
    p.join()


def download_wmts_data(out_path=DATA_DIR_USE, window_x=1, window_y=1):
    layers = [WMTS_LAYERS[-1], WMTS_LAYERS[0]]
    for layer in [layers[0],  layers[-1]]:
        download_layer(layer, out_path, window_x, window_y)


def download_training_data(out_path=DATA_DIR_TRAIN, remove_source=True):
    def show_progress(count, block_size, total_size):
        percent = ((count * block_size) / total_size) * 100
        print('Downloading {0:}: {1:.1f}%'.format(url, percent), end='\r')

    for fn, url in TRAINING_DATA_URLS.items():
        path = out_path + '../' + fn
        urllib.request.urlretrieve(url, path, show_progress)

        with ZipFile(path) as zipf:
            if fn == 'polygons.zip':
                zipf.extract(os.path.basename(JSON_FILE), path=os.path.dirname(path))
            else:
                zipf.extractall(path=out_path)

        if remove_source:
            os.remove(path)

        print()


def download_data():
    # this is a hack to ensure that all data gets downloaded: gdal_translate isn't very stable,
    # so I'll run it 5 times, previously downloaded files don't get overwritten
    for i in range(5):
        try:
            download_wmts_data(window_x=YOLO_IMG_SIZE_X, window_y=YOLO_IMG_SIZE_Y)
        except:
            for i in range(20):
                print('\a')
            continue

    download_training_data()


if __name__ == '__main__':
    download_data()
