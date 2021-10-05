#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5

cd ..
python train_wo_center.py --data-dir datasets/Market-1501 --gpu-id 1 --lr 0.1 \
--interval 10 --model mobilenetv2_x1_0
