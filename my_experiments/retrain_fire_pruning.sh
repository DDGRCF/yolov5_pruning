#! /usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5
cd ..
python train.py --epochs 120 --weights runs/train/exp32/weights/pruning_best_pruning.pt --cfg models/yolov5l.yaml \
--data data/fire.yaml --batch-size 16 --workers 8 --device 1 --retrain --pruning-cfg pruning_cfg.json
cd -
