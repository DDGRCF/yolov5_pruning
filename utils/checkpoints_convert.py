import sys
import argparse
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

import torch
import torch.nn as nn
from models.yolol import Model

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='runs/train/exp39/weights/best.pt')
args = parser.parse_args()
weights_path = Path(args.weights)

model_ckpt = torch.load(weights_path)
model_src = model_ckpt['model']
model_src_dict = model_src.state_dict()
for i, (n, m) in enumerate(model_src.named_modules()):
    if isinstance(m, nn.BatchNorm2d):
        print(i, n, m)
for i, (n, k) in enumerate(model_src.named_parameters()):
    print(f'{i}|{n}|{k.shape}')


