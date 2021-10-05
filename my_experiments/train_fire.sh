#! /usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5
cd ..
python train.py --epochs 300 --weights weights/yolov5l.pt --hyp data/hyps/hyp.finetune.yaml --cfg models/yolov5l.yaml \
--data data/fire.yaml --batch-size 4 --workers 4 --device 3
cd -
