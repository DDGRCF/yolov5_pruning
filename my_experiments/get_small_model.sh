#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5
cd ..
python get_small_model.py --weights runs/train/exp6/weights/best_pruning.pt --cfg models/yolov5l.yaml \
--data data/fire.yaml --device 3
cd -

