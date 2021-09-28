import torch
import torch.nn as nn
import yaml
import logging
import thop
import math
import json
from pathlib import Path
from models.common_pruning import *
from utils.autoanchor import check_anchor_order
from utils.torch_utils import initialize_weights, scale_img, fuse_conv_and_bn, model_info
from utils.plots import feature_visualization
from utils.prune_utils import get_pruning_cfg

logger = logging.getLogger(__name__)
class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class Model(nn.Module):
    def __init__(self, 
                 cfg='models/yolov5l.yaml', 
                 ch=3, nc=None, anchors=None, 
                 pruning_cfg=None):
        super(Model, self).__init__()
        if isinstance(cfg, str):
            with open(cfg) as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = cfg

        if isinstance(pruning_cfg, str) or isinstance(pruning_cfg, Path):
            assert pruning_cfg.match('*.json'), 'this is not pruning cfg'
            with open(pruning_cfg, 'rb') as f:
                self.pruning_cfg = get_pruning_cfg(json.load(f))
        else:
            self.pruning_cfg = pruning_cfg

        ch = self.cfg['ch'] = self.cfg.get('ch', ch)
        if nc and nc != self.cfg['nc']:
            logger.info(f"Overriding model.yaml nc={self.cfg['nc']} with nc={nc}")
            self.cfg['nc'] = nc
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.cfg['anchors'] = round(anchors)
        self.model, self.save = build_model(self.cfg, self.pruning_cfg) 
        self.names = [str(i) for i in range(self.cfg['nc'])]
        self.inplace = self.cfg.get('inplace', True)
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()

        initialize_weights(self)
        
    def forward(self, x, augment=False, profile=False, visualize=False, features=False):
        if augment:
            return self.forward_augment(x)
        return self.forward_once(x, profile, visualize, features)

    def forward_augment(self, x):
        img_size = x.shape[-2: ]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        return torch.cat(y, 1), None

    def forward_once(self, x, profile=False, visualize=False, features=False):
        y, dt = [], []
        z = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            if profile:
                pass
            x = m(x)
            y.append(x if m.i in self.save else None)
            if features:
                z.append(x)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        if features:
            return z
        else:
            return x

    def _descale_pred(self, p, flips, scale, img_size):
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _initialize_biases(self, cf=None):
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

def build_model(cfg, ch_p):
    layers, save = [], []
    for i, (f, n, m, args) in enumerate(cfg['backbone'] + cfg['head']):
        m = eval(m)
        args = [ch_p.get(i), *args[1: ]]
        if m is C3:
            args.insert(1, n)
        elif m is Detect:
            args = [cfg.get('nc'), cfg.get('anchors'), ch_p.get(i)]
        m_ = m(*args)
        np = sum([x.numel() for x in m_.parameters()])
        t = str(m)[8: -2].replace('__main__.', '')
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)

    return nn.Sequential(*layers), sorted(save)
