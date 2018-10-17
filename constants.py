import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR_TRAIN = BASE_DIR + '/data/training/'
DATA_DIR_USE = BASE_DIR + '/data/use/'
JSON_FILE = BASE_DIR + '/data/SolarArrayPolygons.json'

MODEL_DIR = BASE_DIR + '/model/'

SVM_SAVED_MODEL = MODEL_DIR + 'svm/svm_trained.sav'
SVM_SAVED_PREDICTIONS = MODEL_DIR + 'svm/svm_predictions.sav'
SVM_TEST_DATA = MODEL_DIR + 'svm/svm_test_examples.sav'

YOLO_SUBMODULE_DIR = BASE_DIR + '/keras-yolo3/'
YOLO_TRAIN_DATA_DIR = DATA_DIR_TRAIN + 'yolo/'
YOLO_SAVE_DIR = MODEL_DIR + 'yolo/'
YOLO_MODEL = YOLO_SAVE_DIR + 'yolo.h5'
YOLO_PRETRAINED_WEIGHTS = YOLO_SAVE_DIR + 'pretrained.h5'
YOLO_ANCHORS = YOLO_SAVE_DIR + 'anchors.txt'
YOLO_CLASSES = YOLO_SAVE_DIR + 'classes.txt'
YOLO_MIN_CONFIDENCE = 0.55
YOLO_VALIDATION_RATIO = 0.1
YOLO_TRAINING_DATA = DATA_DIR_TRAIN + '../yolo_train.txt'
YOLO_LOG_DIR = BASE_DIR + '/logs/'
YOLO_INPUT_SHAPE = (13 * 32, 13 * 32) # has to be a multiple of 32
YOLO_MAX_EPOCHS = 50
YOLO_BATCH_SIZE = 3
YOLO_LEARNING_RATE = 1e-3
YOLO_PRETRAINED_WEIGHTS_URL = 'https://pjreddie.com/media/files/yolov3.weights'
YOLO_IMG_SIZE_X = 1000
YOLO_IMG_SIZE_Y = 1000

IMG_SIZE_X = 5000
IMG_SIZE_Y = 5000
SVM_WINDOW_SIZE_X = 50
SVM_WINDOW_SIZE_Y = 50

TRAINING_DATA_URLS = {
    'modesto.zip': 'https://ndownloader.figshare.com/articles/3385789/versions/1',
    'stockton.zip': 'https://ndownloader.figshare.com/articles/3385804/versions/1',
    'fresno.zip': 'https://ndownloader.figshare.com/articles/3385828/versions/1',
    'polygons.zip': 'https://ndownloader.figshare.com/articles/3385780/versions/3'
}
WMTS_LAYERS = ['lb', 'lb2016', 'lb2015', 'lb2014']
WMTS_XML = "<GDAL_WMTS>" \
        "<GetCapabilitiesUrl>" \
            "http://maps.wien.gv.at/wmts/1.0.0/WMTSCapabilities.xml" \
        "</GetCapabilitiesUrl>" \
        "<Layer>{:}</Layer>" \
        "<Style>farbe</Style>" \
        "<TileMatrixSet>google3857</TileMatrixSet>" \
        "<DataWindow>" \
            "<UpperLeftX>1800035.8827671</UpperLeftX>" \
            "<UpperLeftY>6161931.920893207</UpperLeftY>" \
            "<LowerRightX>1845677.148953537</LowerRightX>" \
            "<LowerRightY>6123507.385072635</LowerRightY>" \
        "</DataWindow>" \
        "<BandsCount>4</BandsCount>" \
        "<Cache><Depth>2</Depth>" \
        "<Extension>.tif</Extension>" \
        "</Cache>" \
        "<Timeout>600</Timeout>" \
        "<UnsafeSSL>true</UnsafeSSL>" \
        "<ZeroBlockHttpCodes>204,404</ZeroBlockHttpCodes>" \
        "<ZeroBlockOnServerException>true</ZeroBlockOnServerException>" \
      "</GDAL_WMTS>"
