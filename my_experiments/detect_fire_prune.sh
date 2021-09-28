#! /usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5

cd ..
python detect.py --weights runs/train/exp30_0.5p_train/weights/pruning_best_pruning.pt --source https://youtu.be/s-UOhgKVtQs?t=9 \
--use-pruning --device cpu
cd -