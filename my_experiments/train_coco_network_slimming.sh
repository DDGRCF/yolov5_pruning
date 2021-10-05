#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5
cd ..

python train.py --epochs 80 \
--cfg models/yolov5l.yaml --hyp data/hyps/hyp.scratch.yaml --data data/coco.yaml --batch-size 4 \
--device 2 --use-pruning  --pruning-method Network_Slimming \
--s 0.1 --print-sparse-frequency 1 

cd -

