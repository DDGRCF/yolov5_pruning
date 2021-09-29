import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from train import input_size
from models import build_model

test_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.3568, 0.3141, 0.2781], [0.1752, 0.1857, 0.1879])
])

gallery_datasets = datasets.ImageFolder(os.path.join("data", "gallery"),
                                        transform=test_transforms)
query_datasets = datasets.ImageFolder(os.path.join("data", "query"),
                                      transform=test_transforms)

gallery_dataloader = DataLoader(gallery_datasets,
                                batch_size=128,
                                drop_last=False,
                                shuffle=False,
                                num_workers=1)

query_dataloader = DataLoader(query_datasets,
                              batch_size=128,
                              drop_last=False,
                              shuffle=False,
                              num_workers=1)

use_gpu = torch.cuda.is_available()

class_names = gallery_datasets.classes


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)  # flip along w
    return img_flip


def extract_features(model, dataloader):
    features = torch.FloatTensor()
    count = 0
    for data in dataloader:
        img, label = data
        bs, c, h, w = img.size()
        count += bs
        ff = torch.FloatTensor(bs, 96).zero_()  # 2048 if res50
        print(count, end='\r')
        sys.stdout.flush()
        # add two features
        for i in range(2):
            if i == 1:
                img = fliplr(img)
            input_img = Variable(img.cuda())
            # print("=", input_img.shape)
            feature = model(input_img)
            feature = feature.data.cpu()
            # print(ff.shape, feature.shape)
            ff = ff + feature
        # norm features
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff), 0)
    return features


def get_label(img_path):
    labels = []
    for path, _ in img_path:
        filename = os.path.basename(path)
        label = filename.split('_')[0]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(label)
    return labels


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()  #len = 20 得到前20个
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc

    # remove junk index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2
    return ap, cmc


def evaluate(qf, ql, gf, gl):
    query = qf.view(-1, 1)  # query 是一张图
    score = torch.mm(gf, query)  # 计算得分[1, num]
    score = score.squeeze(1).cpu()
    score = score.numpy()
    #predict index
    index = np.argsort(score)
    index = index[::-1]  # index 倒过来
    # 得到前20个
    # index = index[0:20]

    # good index , label一致
    good_index = np.argwhere(gl == ql)
    # print("good_index", gl, '\n', ql, gl == ql, type(gl))
    junk_index = np.argwhere(gl == "bg")

    CMC = compute_mAP(index, good_index, junk_index)
    return CMC


if __name__ == "__main__":
    parser = argparse.ArgumentParser('help')
    parser.add_argument('--weight_path',
                        type=str,
                        default="./checkpints/last.pt")
    parser.add_argument("--model", type=str, default="mudeep")
    args = parser.parse_args()

    model = build_model(name=args.model, num_classes=len(class_names))
    assert os.path.isfile(
        "./checkpoint/%s/%s_last.pt" %
        (args.model, args.model)), "Error: no checkpoint file found!"
    print('Loading from checkpoint/last.pt')
    checkpoint = torch.load("./checkpoint/%s/%s_last.pt" %
                            (args.model, args.model))
    net_dict = checkpoint['net_dict']
    model.load_state_dict(net_dict)

    model.eval()
    if use_gpu:
        model = model.cuda()

    gallery_features = extract_features(model, gallery_dataloader)
    query_features = extract_features(model, query_dataloader)

    gallery_label = np.array(get_label(gallery_datasets.imgs))
    query_label = np.array(get_label(query_datasets.imgs))

    if use_gpu:
        gallery_features = gallery_features.cuda()
        query_features = query_features.cuda()

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_features[i], query_label[i],
                                   gallery_features, gallery_label)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        # print(i, ":",ap_tmp)
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / len(query_label)

    print("\tRank@1:%f\n\tRank@5:%f\n\tRank@10:%f\n\tmAP:%f" %
          (CMC[0], CMC[4], CMC[9], ap / len(query_label)))
