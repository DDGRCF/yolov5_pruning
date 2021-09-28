import torch
import torch.nn as nn
from utils.general import non_max_suppression

_all_ = ['Conv', 'Bottleneck', 'C3', 'SPP', 'Focus', 'Concat', 'NMS']

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c_p, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        c_p = c_p[0] if isinstance(c_p, list) else c_p
        c1 = c_p[0][1]
        c2 = c_p[0][0]
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c_p1, c_p2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        self.add = shortcut
        self.cv1 = Conv(c_p1, 1, 1)
        self.cv2 = Conv(c_p2, 3, 1, g=g)
        self.keep_index = torch.LongTensor(c_p2[-1])
        
    def forward(self, x):
        if self.add and not self.training:
            x.index_add_(1, self.keep_index.to(x.device), self.cv2(self.cv1(x)))
        elif self.add and self.training:
            x = x.index_add(1, self.keep_index.to(x.device), self.cv2(self.cv1(x)))
        else:
            x = self.cv2(self.cv1(x))
        return x

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c_p, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        self.cv1 = Conv(c_p[0], 1, 1)
        self.cv2 = Conv(c_p[1], 1, 1)
        self.m = nn.Sequential(
            *[Bottleneck(c_p[2 + i * 2], c_p[3 + i * 2], shortcut, g, e=1.0) for i in range(n)]
            )
        self.cv3 = Conv(c_p[-1], 1)  
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, T, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c_p, k=(5, 9, 13)):
        super(SPP, self).__init__()
        self.cv1 = Conv(c_p[0], 1, 1)
        self.cv2 = Conv(c_p[1], 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c_p, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c_p, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))

class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class
    max_det = 1000  # maximum number of detections per image

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], self.conf, iou_thres=self.iou, classes=self.classes, max_det=self.max_det)

