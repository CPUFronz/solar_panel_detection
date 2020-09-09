import os
import sys
import pickle
import urllib.request
from glob import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from osgeo import gdal
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping

import utils
from constants import JSON_FILE
from constants import DATA_DIR_USE
from constants import DATA_DIR_TRAIN
from constants import IMG_SIZE_X
from constants import IMG_SIZE_Y
from constants import YOLO_IMG_SIZE_X
from constants import YOLO_IMG_SIZE_Y
from constants import YOLO_SUBMODULE_DIR
from constants import YOLO_PRETRAINED_WEIGHTS_URL
from constants import YOLO_TRAIN_DATA_DIR
from constants import YOLO_MODEL
from constants import YOLO_ANCHORS
from constants import YOLO_CLASSES
from constants import YOLO_LOG_DIR
from constants import YOLO_SAVE_DIR
from constants import YOLO_TRAINING_DATA
from constants import YOLO_PRETRAINED_WEIGHTS
from constants import YOLO_INPUT_SHAPE
from constants import YOLO_MIN_CONFIDENCE
from constants import YOLO_VALIDATION_RATIO
from constants import YOLO_BATCH_SIZE
from constants import YOLO_LEARNING_RATE
from constants import YOLO_MAX_EPOCHS


sys.path.append(YOLO_SUBMODULE_DIR)
from yolo import YOLO
from yolo3.utils import letterbox_image
from yolo3.utils import get_random_data
from yolo3.model import yolo_body
from yolo3.model import yolo_loss
from yolo3.model import preprocess_true_boxes


class YoloWrapper(YOLO):
    def __init__(self):
        self._defaults['model_path'] = YOLO_MODEL
        self._defaults['anchors_path'] = YOLO_ANCHORS
        self._defaults['classes_path'] = YOLO_CLASSES
        self._defaults['score'] = YOLO_MIN_CONFIDENCE
        self.__dict__.update(self._defaults)

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()

        if os.path.exists(YOLO_MODEL) and os.path.getsize(YOLO_MODEL) > 1 * 1024:
            self.boxes, self.scores, self.classes = self.generate()

    def _initialize_training(self):
        if not os.path.exists(YOLO_TRAINING_DATA):
            polygons = utils.load_polygons(JSON_FILE)
            tif_files = glob(DATA_DIR_TRAIN + '*.tif')
            self._create_examples(tif_files, polygons)

        if not os.path.exists(YOLO_PRETRAINED_WEIGHTS):
            self._get_pretrained_weights()

        self.class_names = self._get_class()
        self.num_classes = len(self.class_names)
        self.anchors = self._get_anchors()
        self.num_anchors = len(self.anchors)

        self.model = self._create_model()
        self.max_iterations = YOLO_MAX_EPOCHS

        self.log = TensorBoard(YOLO_LOG_DIR + 'yolo_{:04d}_iterations'.format(self.max_iterations))
        self.checkpoints = ModelCheckpoint(YOLO_SAVE_DIR + 'yolo_weights.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_weights_only=True, save_best_only=True, period=2)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        self.early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1)

    def _get_pretrained_weights(self, url=YOLO_PRETRAINED_WEIGHTS_URL, fn=YOLO_SAVE_DIR+'yolov3_darknet.weights'):
        def show_progress(count, block_size, total_size):
            percent = ((count * block_size) / total_size) * 100
            print('Downloading {0:}: {1:.1f}%'.format(url, percent), end='\r')

        urllib.request.urlretrieve(url, fn, show_progress)

        convert = YOLO_SUBMODULE_DIR + 'convert.py'
        args = '-w {0:} {1:} {2:}'.format(YOLO_SUBMODULE_DIR + 'yolov3.cfg', fn, YOLO_PRETRAINED_WEIGHTS)
        command = 'python {0:} {1:}'.format(convert, args)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.system(command)

    def _create_model(self):
        K.clear_session()
        input_image = Input(shape=(None, None, 3))
        h, w = YOLO_INPUT_SHAPE

        y_true = [
            Input(shape=(h // 32, w // 32, self.num_anchors // 3, self.num_classes + 5)),
            Input(shape=(h // 16, w // 16, self.num_anchors // 3, self.num_classes + 5)),
            Input(shape=(h // 8, w // 8, self.num_anchors // 3, self.num_classes + 5))
        ]

        model_body = yolo_body(input_image, self.num_anchors // 3, self.num_classes)
        model_body.load_weights(YOLO_PRETRAINED_WEIGHTS, by_name=True, skip_mismatch=True)

        args = {'anchors': self.anchors, 'num_classes': self.num_classes, 'ignore_thresh': 0.5}
        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss', arguments=args)([*model_body.output, *y_true])
        model = Model([model_body.input, *y_true], model_loss)

        return model

    def _create_examples(self, tif_files, polygons, split_images=True, split_path=YOLO_TRAIN_DATA_DIR, step_x=YOLO_IMG_SIZE_X, step_y=YOLO_IMG_SIZE_Y):
        lines = defaultdict(list)

        if not os.path.exists(split_path):
            os.mkdir(split_path)

        for tif in tif_files:
            src_ds = gdal.Open(tif)

            for pv in polygons:
                # only create a list of pvs for the current file, saves a lot of time
                if os.path.basename(tif).replace('.tif', '') == pv.filename:
                    # append only if center is not _NaN_
                    if type(pv.center_x) != str or type(pv.center_y) != str:
                        if split_images:
                            for win_x in range(0, IMG_SIZE_X, step_x):
                                for win_y in range(0, IMG_SIZE_Y, step_y):
                                    if win_x <= pv.center_x <= win_x + step_x and win_y <= pv.center_y <= win_y + step_y:
                                        new_fn = split_path + os.path.basename(tif) + '_' + str(win_x) + '_' + str(win_y)

                                        if not os.path.exists(new_fn):
                                            new_ds = gdal.Translate(new_fn, src_ds, srcWin=[win_x, win_y, step_x, step_y])
                                            del new_ds

                                        new_center_x = pv.center_x - win_x
                                        new_center_y = pv.center_y - win_y
                                        x_min = int(new_center_x - pv.width / 2)
                                        y_min = int(new_center_y - pv.height / 2)
                                        x_max = int(new_center_x + pv.width / 2)
                                        y_max = int(new_center_y + pv.height / 2)
                                        lines[os.path.abspath(new_fn)].append('{},{},{},{},0 '.format(x_min, y_min, x_max, y_max))
                        else:
                            x_min = int(pv.center_x - pv.width / 2)
                            y_min = int(pv.center_y - pv.height / 2)
                            x_max = int(pv.center_x + pv.width / 2)
                            y_max = int(pv.center_y + pv.height / 2)
                            lines[os.path.abspath(tif)].append('{},{},{},{},0 '.format(x_min, y_min, x_max, y_max))

        with open(YOLO_TRAINING_DATA, 'w') as f:
            for key, values in lines.items():
                line = key + ' '
                for value in values:
                    line += value
                f.write(line + '\n')

    def _generate_data(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        n = len(annotation_lines)
        if n == 0 or batch_size <= 0: return None

        i = 0
        while True:
            image_data = []
            box_data = []
            for b in range(batch_size):
                if i == 0:
                    np.random.shuffle(annotation_lines)
                image, box = get_random_data(annotation_lines[i], input_shape, random=True)
                image_data.append(image)
                box_data.append(box)
                i = (i + 1) % n
            image_data = np.array(image_data)
            box_data = np.array(box_data)
            y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
            yield [image_data, *y_true], np.zeros(batch_size)

    def train(self, training_data_path=YOLO_TRAINING_DATA, batch_size=YOLO_BATCH_SIZE):
        self._initialize_training()

        with open(training_data_path, 'r') as f:
            examples = f.readlines()

        np.random.shuffle(examples)
        num_examples = len(examples)
        num_val = int(num_examples * YOLO_VALIDATION_RATIO)
        num_train = num_examples - num_val
        examples[:num_train], examples[num_train:]

        self.model.compile(optimizer=Adam(lr=YOLO_LEARNING_RATE), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        self.model.fit_generator(self._generate_data(examples[:num_train], batch_size, YOLO_INPUT_SHAPE, self.anchors, self.num_classes),
                                 steps_per_epoch=max(1, num_train//batch_size),
                                 validation_data=self._generate_data(examples[num_train:], batch_size, YOLO_INPUT_SHAPE, self.anchors, self.num_classes),
                                 validation_steps=max(1, num_val//batch_size),
                                 epochs=YOLO_MAX_EPOCHS,
                                 initial_epoch=0,
                                 callbacks=[self.log, self.checkpoints, self.reduce_lr, self.early_stopping])
        self.model.save_weights(YOLO_MODEL)

        self.close_session()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def predict(self, filename, show=False):
        image = Image.open(filename)

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, _ = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), filename))

        out_boxes = [[b[1], b[0], b[3], b[2]] for b in out_boxes]
        if len(out_boxes) > 0 and show:
            plt.imshow(image)
            for i, box in enumerate(out_boxes):
                box = [int(b) for b in box]
                rect_x, rect_y = utils.calc_rectangle(x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3])
                plt.plot(rect_x, rect_y, color='red')
                plt.text(box[3], box[2], '{:.2f}'.format(out_scores[i]), color='red')

            plt.show()

        return filename, out_boxes, out_scores


if __name__ == '__main__':
    yolo = YoloWrapper()
    if not os.path.exists(YOLO_MODEL):
        yolo.train()
    results = []
    for tif in glob(DATA_DIR_USE + '*.tif'):
        results.append(yolo.predict(tif))
    yolo.close_session()

    with open(YOLO_SAVE_DIR + 'yolo_results.pickle', 'wb') as f:
        pickle.dump(results, f)
