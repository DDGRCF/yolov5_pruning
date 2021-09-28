#! /bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate yolov5 
cd ..
python get_sparse_model.py --weights runs/train/exp35/weights/best_sparse.pt \
--data data/fire.yaml --device 1