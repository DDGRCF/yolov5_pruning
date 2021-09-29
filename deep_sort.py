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
from deep_sort import DeepSort

@torch.no_grad()
def run(opt):
    weights, deep_weights, device, use_pruning, max_dist = \
    opt.weights, opt.deep_weights, opt.device, opt.use_pruning, opt.max_dist

    device = select_device(device)
    if use_pruning:
        print("Using pruning model")
    else:
        print("Using normal model")
    model = attempt_load(weights, map_location=device, pruning=use_pruning)
    model.eval()
    deepsort = DeepSort(deep_weights, max_dist)
    stride = int(model.stride.max())
    names = model.names
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Camera can not open')
        exit()
    while True:
        ret, img0 = cap.read()
        if not ret:
            print('Get the pictures failed')
            break
        img0, img = Image_deal(img0, 640, stride)
        img = torch.from_numpy(img).float().unsqueeze(0).to(device)
        img /= 255.0
        pred = model(img)[0]
        det = non_max_suppression(pred, 0.25, 0.45)[0]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2: ], det[:, :4], img0.shape).round()
            # TODO: to be all tensor calculate
            det = det.cpu().numpy()
            bbox_xyxy, cls_conf, cls_id = np.split([4, 1, 1], axis=-1)
            bbox_cxcywh = xyxy2xywh(bbox_xyxy)
            det_mot = deepsort.update(bbox_cxcywh, cls_conf, cls_id, img0)
            if len(det_mot) > 0:
                bbox_xyxy, track_ids, cls_ids = det_mot[:, :4], det_mot[:, 4], det_mot[:, -1]
                draw_bboxes(img0, bbox_xyxy, track_ids=track_ids, cls_ids=cls_ids, offset=(0, 0), names=names)
        cv2.imshow('MOT', img0)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def Image_deal(img0, img_size, stride):
    img = letterbox(img0, img_size, stride)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    return img0, img

def draw_bboxes(img, bboxes, track_ids=None, cls_ids=None, offset=(0, 0), names=None):
    bboxes = bboxes.astype(np.int32)
    offsets = np.tile(np.array(offset, dtype=np.float32)[np.newaxis, :], 2)
    bboxes += offsets
    for i, bbox in enumerate(bboxes):
        track_id = int(track_ids[i]) if track_ids is not None else 0
        cls_id = int(cls_ids[i]) if cls_ids is not None else 0
        name = names[cls_id] if names is not None else cls_id
        label = "{}|{:d}".format(name, track_id)
        plot_one_box(bbox, img, label=label, color=colors(cls_id, True), line_thickness=2)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=str, default='/runs/train/train_weights.pth')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--use-pruning', action='store_true')

    opt = parser.parse_args()
    return opt

def main(opt):
    run(opt)

if __name__ == '__main__':
    opt = get_opt()
    main(opt)