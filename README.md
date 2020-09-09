# Yolo 3 for Solar Panel Detection

## Introduction
A proof-of-concept implementation of using YOLOv3 for detecting solar panels in high resolution aerial images.

## Usage

First run `python download.py` to download training data. Afterwards start the training by launching `python main.py`.

**Important:** You have to adjust the paths in [data/yolo_train.txt](data/yolo_train.txt), so that the image files match your path.

## Acknowledgement
YOLOv3 code taken from https://github.com/qqwweee/keras-yolo3
