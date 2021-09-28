#! /usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5
cd ..
python detect.py --weights weights/yolov5l.pt \
--source data/images --device 1 --save-txt
cd -