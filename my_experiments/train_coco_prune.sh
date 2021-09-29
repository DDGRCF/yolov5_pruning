#! /bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5
cd ..
python train.py --epochs 12 --weights weights/yolov5l.pt --hyp data/hyps/hyp.finetune.yaml --cfg models/yolov5l.yaml \
--data data/coco.yaml --batch-size 2 --workers 2 --device 3 --use-pruning --layer-rate 0.6  \
--layer-gap 0,321,3 --skip-downsample --pruning-method SFP 
cd -
