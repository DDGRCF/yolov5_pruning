'''
这个脚本是即为SFP服务，又为Network Slimming服务
当为SFP运行时，作为主文件
当为Network Slimming服务时，作为副文件
'''
import yaml
import json
import torch

import argparse
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from typing import OrderedDict

from models.yolo import Model as Model
from models.yolov5_pruning import Model as Model_Pruning

from utils.general import set_logging
from utils.prune_utils import model_compare, model_eval, get_mask, setup_seed
from utils.torch_utils import select_device

setup_seed(20)
set_logging()

logger = logging.getLogger(__name__)

@torch.no_grad()
def pruning(weights, 
            device,
            cfg,
            data,
            pruning_path_dir
            ):
    # Pre Works
    network_slimming = False
    device = select_device(device) if isinstance(device, str) else device
    if isinstance(weights, str):
        weights = Path(weights)
        assert weights.match('*.pt'), 'the file must be the type of *.pt '
        ckpt = torch.load(weights, map_location=lambda storage, loc: storage)
        model = ckpt['model'].float() # fp16 -> fp32
    else:
        model = weights
        network_slimming = True
    if pruning_path_dir == '':
        pruning_weights = weights.parents[0] / ('pruning_' + weights.name)
        pruning_cfg_path = weights.parents[0] / 'pruning_cfg.json'
    else:
        pruning_path_dir = Path(pruning_path_dir)
        assert pruning_path_dir.is_dir(), 'the pruning path dir must be dir'
        pruning_weights = pruning_path_dir / 'pruning_sparse.pt'
        pruning_cfg_path = pruning_path_dir / 'pruning_cfg.json'
    if isinstance(data, str):
        data = Path(data)
        assert data.match('*.yaml'), 'the file must be the type of *.yaml'
        with open(data) as f:
            data_dict = yaml.safe_load(f)
    else:
        data_dict = data
        
    cfg = Path(cfg)
    assert cfg.match('*.yaml'), 'the file must be the type of *.yaml'
    with open(cfg) as f:
        cfg = yaml.safe_load(f)

    # Load the model configuration
    cfg['nc'] = model.yaml.get('nc', 1)

    input_from = OrderedDict()
    shortcut = OrderedDict()

    # get orginal model configuration
    for i, m in enumerate(cfg['backbone'] + cfg['head']):
        input_from[i] = m[0]
        if m[2] == 'C3':
            shortcut[i] = m[-1][-1]
        else:
            shortcut[i] = None   

    # parameters TODO:to be general
    first_layer = 0
    last_layer = 641
    inter_layer = 6

    keep_mask = []
    bn_order = []
    layer_index = [i for i in range(first_layer, last_layer + 1, inter_layer)]
    state_dict = model.state_dict()
    channel_index = OrderedDict()
    state_dict_ = OrderedDict()
    bn_module = OrderedDict()
    conv_module = OrderedDict()
    bn_activation = OrderedDict()

    # Store the bn and conv module
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, nn.BatchNorm2d):
            module_ = deepcopy(module)
            module_.eval()
            bn_module[name] = module_
        elif isinstance(module, nn.Conv2d):
            module_ = deepcopy(module)
            module_.eval()
            conv_module[name] = module_  

    # Channel pruning
    len_pbar = len(state_dict)
    pbar = tqdm(enumerate(state_dict.items()), total=len_pbar)
    for i, (n, param) in pbar:
        m_index = int(n.split('.')[1])
        if i in layer_index: 
        # pruning input channels
            before_i = i - inter_layer
            if before_i > -1:
                if '.cv1.' in n and '.m.' not in n:
                    f = input_from[m_index - 1]
                    if isinstance(f, list):
                        k1 = keep_mask[f[0]][-1]
                        k2 = channel_index[f[-1]][-1][-1] + keep_mask[f[0]][0][0]
                        k = torch.cat((k1, k2), 0)
                        # bn value
                        bn_act1 = bn_order[f[0]]
                        bn_act2 = bn_activation[f[-1]][-1]
                        bn_act_total = torch.cat((bn_act1, bn_act2), 0)
                    else:
                        k = keep_mask[f][-1]
                        # bn value
                        bn_act_total = bn_order[-1]
                elif ('cv2' in n) and (m_index == 8) and '.m.' not in n: # 处理 Spp的cv2
                    k = torch.cat([(keep_mask[-1][-1] + d * keep_mask[-1][0][0]) for d in range(4)]) # TODO:4 to general
                    # bn value
                    bn_act_total = torch.cat([bn_order[-1] for _ in range(4)])
                elif ('cv2' in n) and (m_index != 8) and '.m.' not in n:
                    f = input_from[m_index - 1]
                    if isinstance(f, list):
                        k1 = keep_mask[f[0] - 1][-1]
                        k2 = channel_index[f[-1]][-1][-1] + keep_mask[f[0] - 1][0][0]
                        k = torch.cat((k1, k2), 0)
                        # bn value 
                        bn_act1 = bn_order[f[0] - 1]
                        bn_act2 = bn_activation[f[-1]][-1]
                        bn_act_total = torch.cat((bn_act1, bn_act2), 0)
                    else:
                        k = keep_mask[f - 1][-1]
                        # bn value
                        bn_act_total = bn_order[f - 1]
                elif 'cv3' in n and '.m.' not in n:
                    if shortcut[m_index] == True:
                        k1 = torch.LongTensor(range(keep_mask[-1][0][0]))
                    else:
                        k1 = keep_mask[-1][-1]
                    k2 = channel_index[m_index][1][-1] + keep_mask[-1][0][0]
                    k = torch.cat((k1, k2), 0)
                    # bn value
                    bn_act1 = bn_order[-1]
                    bn_act2 = bn_activation[m_index][1]
                    bn_act_total = torch.cat((bn_act1, bn_act2), 0)
                elif '.cv1.' in n and shortcut[m_index] is not None and '.m.' in n:  # 处理m模块的第一个输入的从cv1
                    if '.0.' in n:
                        k = keep_mask[-2][-1]
                        # bn value
                        bn_act_total = bn_order[-2]
                    else:
                        if shortcut[m_index]:
                            k = torch.LongTensor(range(keep_mask[-1][0][0]))
                            # bn value
                        elif not shortcut[m_index]:
                            k = keep_mask[-1][-1]
                        bn_act_total = bn_order[-1] 
                else: # 处理m模块的cv2和一般模块
                    k = keep_mask[-1][-1]
                    # bn value
                    bn_act_total = bn_order[-1]
                state_dict_[n] = torch.index_select(param, 1, k)
                conv_name = n.replace('conv.weight', 'conv')
                conv_sum = conv_module[conv_name].weight.data.sum((2, 3))
                offset = conv_sum.matmul(bn_act_total.view(-1, 1)).view(-1)

            # pruning input channels
            temp_param = state_dict_[n] if before_i > -1 else param
            if shortcut[m_index] == True and '.cv1.' in n and '.m.' not in n:  
                keep_index = torch.LongTensor(range(temp_param.shape[0]))
            else:
                keep_index = torch.nonzero(torch.sum(temp_param.data, dim=(1, 2, 3)) != 0).view(-1) 
            state_dict_[n] = torch.index_select(temp_param, 0, keep_index)
            keep_mask.append((temp_param.shape[: 2], keep_index))
            if m_index in channel_index:
                channel_index[m_index] += [(temp_param.shape[: 2], keep_index)]
            else:
                channel_index[m_index] = [(temp_param.shape[: 2], keep_index)]
        elif (i not in layer_index) and (first_layer < i < last_layer + 1):
            # update running_mean of BN
            if 'running_mean' in n:
                bn_name = n.replace('.running_mean', '')
                bn_m = bn_module[bn_name]
                if before_i > -1:
                    state_dict_[n] = param  - offset
                # bn value
                if network_slimming: # 由于network slimming中的weights接近为0，因此只考虑bias
                    bn_act = F.silu(bn_m.bias.data).view(-1)
                else:
                    zero_input = torch.zeros(1, bn_m.weight.data.shape[0], 1, 1)
                    bn_act = bn_m(zero_input).view(-1)
                    bn_act = F.silu(bn_act, inplace=True)
                mask = get_mask(keep_index, bn_m.weight.data.shape[0])
                bn_act *= mask
                bn_order.append(bn_act)
                if m_index in bn_activation:
                    bn_activation[m_index] += [bn_act]
                else:
                    bn_activation[m_index] = [bn_act]
            # others
            if len(param.shape) != 0:
                temp_param = state_dict_[n] if before_i > -1 and 'running_mean' in n else param
                state_dict_[n] = torch.index_select(temp_param, 0, keep_mask[-1][-1])
            else:
                state_dict_[n] = param.clone()
        else:
            if 'weight' in n:
                f = input_from[m_index]
                k = channel_index[f[int((i - last_layer - 3) / 2)]][-1][-1]
                state_dict_[n] = torch.index_select(param, 1, k)
                # bn value
                bn_act_total = bn_activation[f[int((i - last_layer - 3) / 2)]][-1]
                conv_name = n.rstrip('.weight')
                conv_sum = conv_module[conv_name].weight.data.sum((2, 3))
                offset = conv_sum.matmul(bn_act_total.view(-1, 1)).view(-1)
            elif 'bias' in n:
                state_dict_[n] = param + offset
            else:
                state_dict_[n] = param.clone()      

    # 保存裁剪的配置
    logger.info('the channels pruning is completed!!!')
    pruning_cfg = OrderedDict()
    index_ = -1
    for i, (n, param) in enumerate(state_dict_.items()):
        m_index = int(n.split('.')[1])
        shape = list(param.shape[: 2])
        if i in layer_index:
            index_ += 1
            if m_index in pruning_cfg:
                pruning_cfg[m_index] += [(shape, keep_mask[index_][-1].tolist())]
            else:
                pruning_cfg[m_index] = [(shape, keep_mask[index_][-1].tolist())]
        elif i > last_layer:
            if 'anchor' not in n:
                if 'weight' in n:
                    if m_index in pruning_cfg:
                        pruning_cfg[m_index] += [shape[1]]
                    else:
                        pruning_cfg[m_index] = [shape[1]]

    # Save the pruning configuration
    with open(pruning_cfg_path, 'w') as f:
        f.write(json.dumps(pruning_cfg))
    
    # Init small model
    model_pruning = Model_Pruning(cfg, nc=1, pruning_cfg=pruning_cfg)

    # Load the parameters into small model
    new_state_dict = OrderedDict()
    for i, (name, param) in enumerate(state_dict_.items()):
        new_state_dict[name] = state_dict_[name]

    model_pruning.load_state_dict(new_state_dict, strict=False)
    model_pruning.names = model.names
    # Evaluate the model
    model_compare(model, model_pruning, mode='features')
    
    model_eval(model, data_dict, device)
    model_eval(model_pruning, data_dict, device)

    # Save the model
    if network_slimming:
        ckpt = {
            'model':deepcopy(model_pruning.half()),
            'best_fitness': 0,
            'epoch': -1
        }
    else:
        ckpt['model'] = deepcopy(model_pruning.half())
        ckpt['best_fitness'] = 0
        ckpt['epoch'] = -1
    torch.save(ckpt, pruning_weights)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5l.pt')
    parser.add_argument('--device', type=str, default='2')
    parser.add_argument('--cfg', type=str, default='models/yolov5l.yaml')
    parser.add_argument('--data', type=str, default='data/fire.yaml')
    parser.add_argument('--pruning_path_dir', type=str, default='')
    opt = parser.parse_args()
    return opt

def main(opt):
    pruning(**vars(opt))

if __name__ == '__main__':
    opt = get_opt()
    main(opt)