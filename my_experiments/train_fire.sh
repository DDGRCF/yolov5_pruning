#! /usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5
cd ..
python train.py --epochs 200 --weights weights/yolov5l.pt --cfg models/yolov5l.yaml \
--data data/fire.yaml --batch-size 10 --workers 8 --device 1
cd -