#! /usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5
cd .. 
python detect.py --weights runs/train/exp6/weights/last.pt \
--source datasets/fire/images/val/ --device 3
