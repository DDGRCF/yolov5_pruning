#! /usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5
cd ..
python detect.py --weights weights/yolov5l_new.pt \
--source data/images --device 1 --reconstruct --save-txt
cd -