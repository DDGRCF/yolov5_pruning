#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5

cd ../runs/train
logdir=${1}
tensorboard --logdir ${logdir}
cd -

