#! /usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5
cd ..
python train.py --weights weights/yolov5l.pt --cfg models/yolov5l.yaml \
--batch-size 8 --device 0 --layer-rate 0.8 --pruning-frequency 1 \
--use-pruning --skip-downsample
cd -