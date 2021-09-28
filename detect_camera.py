import cv2
import torch
import argparse
import torch.nn as nn
import numpy as np
from models.experimental import attempt_load
from utils.torch_utils import select_device 
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box, colors
@torch.no_grad()
def run(opt):
    weights, data, cfg, device, use_pruning = opt.weights, opt.data, opt.cfg, opt.device, opt.use_pruning
    device = select_device(device)
    model = attempt_load(weights, map_location=device, pruning=True)
    model.eval()
    stride = int(model.stride.max())
    names = model.names
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('error1')
        exit()
    while True:
        ret, img0 = cap.read()
        if not ret:
            print('error2')
            break
        
        img0, img = Image_deal(img0, 640, stride)
        img = torch.from_numpy(img).float().unsqueeze(0).to(device)
        img /= 255.0
    
        pred = model(img)[0]
        det = non_max_suppression(pred, 0.25, 0.45)[0]
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2: ], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = names[c]
                plot_one_box(xyxy, img0, label=label, color=colors(c, True), line_thickness=3)
        pred = non_max_suppression(pred, 0.25, 0.45)
        cv2.imshow('IMAGE', img0)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def Image_deal(img0, img_size, stride):
    img = letterbox(img0, img_size, stride)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    return img0, img

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/media/r/文档/CV-Works/CV-Scripts/Yolov5_Pruning/yolov5_Chinese/runs/train/exp30_0.5p_train/weights/pruning_best_pruning.pt')
    parser.add_argument('--data', type=str, default='data/fire.yaml')
    parser.add_argument('--cfg', type=str, default='models/yolov5l.yaml')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--use-pruning', action='store_true')

    opt = parser.parse_args()
    return opt

def main(opt):
    run(opt)

if __name__ == '__main__':
    opt = get_opt()
    main(opt)