#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5
# --hyp hyp.finetune.yaml --weights weights/yolov5l.pt
cd ..
python train.py --epochs 300 \
--cfg models/yolov5l.yaml --data data/fire.yaml --batch-size 16 \
--device 2 --use-pruning --skip-list 0 3 --pruning-method Network_Slimming \
--s 1 --print-sparse-frequency 2 --s_span 10 100 --warm_up_epoch 66
cd -

