# YOLOv5 Pruning

 ![](https://img.shields.io/badge/release-1.0.0-red)

介绍

这个项目是自己在暑期学习模型剪枝和YOLOv5写的，其中存在许多不足的之处还待继续完善

本项目涉及的算法如下：

1. YOLOv5的GITHUB地址：https://github.com/ultralytics/yolov5.git

2. SFP的Paper地址：https://arxiv.org/abs/1808.07471

3. Network Slimming的Paper地址：https://arxiv.org/abs/1708.06519

本仓库包含一下内容：

1. 如何安装使用YOLOv5 Pruning
2. 如何使用SFP进行剪枝
3. 如何使用Network Slimming进行剪枝

## 内容列表

- [安装](#安装)
- [使用说明](#使用说明)
  - [SFP](#SFP)
  - [Network_Slimming](#Network_Slimming)
  - [检测](#检测)

- [相关仓库](#相关仓库)

## 安装

创建虚拟环境

```shell
$ create -n yolov5 python=3.7 -y
```

克隆仓库

```shell
$ git clone https://github.com/DDGRCF/yolov5_Chinese.git
```

`cd yolov5_Chinese `安装依赖

```shell
$ pip install -r requirements.txt
```

在该目录下创建一个`datasets`文件夹，该文件夹用来存放各种数据集（你可以创建一个数据集的软链接）

本项目由于条件限制，只使用火焰数据集（`fire`），其配置文件在`data\fire.yaml`。该数据集已上传网盘。

## 使用说明

### SFP

#### 训练

```shell
$ python train.py --epochs 220 --weights weights/yolov5l.pt --hyp hyp.finetune.yaml --cfg models/yolov5l.yaml --data data/fire.yaml --batch-size 16 --workers 8 --device 3 --use-pruning --layer-rate 0.4 --layer-gap 0,321,3 --skip-downsample --pruning-method SFP  --skip-list 0 3
:<<! 
其中use-pruning代表进行剪枝，layer-rate代表剪枝率（1为不剪枝），skip-list代表不需要剪枝的层（只能是0 3 6...， pruning-method代表使用剪枝的方法）(这个暂时不可用，如果启用请将utils、prune_utils.py的76行注释取消)
!
```

#### 裁剪

```shell
$ python get_small_model.py --weights runs/train/exp*/weights/best_pruning.pt --cfg models/yolov5l.yaml --data data/fire.yaml --device 0
```

### Network_Slimming

#### 训练

这个目前有点问题，原始算法的脚本和一些YOLOv3的剪枝脚本，BN scale仅需`1.0e-5~1.0e-2`左右，但我这里在不加载预训练模型，且把模型的BN初始化为`0.5`时，BN scale需要差不多`10~100`其BN weights才进行明显的稀疏，而加载预训练模型后则需要差不多`100~1000`才能进行明显的稀疏。

```shell
$ python train.py --epochs 300 --cfg models/yolov5l.yaml --data data/fire.yaml --batch-size 16 --device 0 --use-pruning --skip-list 0 3 --pruning-method Network_Slimming \
--s 1 --print-sparse-frequency 2 --s_span 10 100 --warm_up_epoch 66
:<<! 
其中 pring-sparse-frequency代表进行多少次epoch输出稀疏度直方图，直方图在`runs\train\exp*\weights\histogram`中，warm_up_epoch是我为了进行快速稀疏而对前n个epoch的scale进行方法，s代表scale
!
```

#### 裁剪

```shell
$ python get_sparse_model.py --weights runs/train/exp*/weights/best_sparse.pt --data data/fire.yaml --device 0 --ratio 0.70
:<<!
其中ratio代表裁剪率
!
```

### 检测

```shell
python detect.py --weights runs/train/exp*/weights/pruning_*.pt --source data/images/fire --use-pruning
```

## 相关仓库

- [YOLOv5](https://github.com/ultralytics/yolov5.git)

- [[Lam1360](https://github.com/Lam1360)/**[YOLOv3-model-pruning](https://github.com/Lam1360/YOLOv3-model-pruning)**]

