#! /usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5
cd ..
python train.py --weights weights/yolov5l.pt --cfg models/yolov5l.yaml --data data/coco128.yaml \
--batch-size 8 --workers 4 --device 0