#! /usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5

cd ..
python detect.py --weights runs/train/exp6/weights/pruning_best_pruning.pt --source datasets/fire/images/val/ --use-pruning --device 3
cd -
