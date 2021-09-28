#! /usr/bin/env bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5
cd ..
python train.py --epochs 220 --weights weights/yolov5l.pt --hyp hyp.finetune.yaml --cfg models/yolov5l.yaml \
--data data/fire.yaml --batch-size 16 --workers 8 --device 3 --use-pruning --layer-rate 0.4  \
--layer-gap 0,321,3 --skip-downsample --pruning-method SFP  --skip-list 0 3
cd -
