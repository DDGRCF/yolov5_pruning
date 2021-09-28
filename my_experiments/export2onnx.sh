#! /usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5

cd ..
python export.py --weights weights/yolov5l.pt