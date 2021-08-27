# MMSegmentation - SFNet

基于 [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 的 SFNet 复现，提交于 [OpenMMLab 算法生态挑战赛](https://openmmlab.com/competitions/algorithm-2021)。

原始论文：[Semantic Flow for Fast and Accurate Scene Parsing](https://arxiv.org/abs/2002.10120)

阅读其他语言的版本：[English](./README_en-US.md) | 简体中文

## 目录

- [环境要求](#prerequisites)
- [目录结构](#file-structure)
- [用法](#usage)
    - [训练](#train)
    - [测试](#test)
- [结果](#results)
- [引用](#citation)

## <a name="prerequisites"></a> 环境要求

- [mim](https://github.com/open-mmlab/mim) == v0.1.0
- [mmcv-full](https://github.com/open-mmlab/mmcv) == v1.3.8
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) == v0.15.0

```shell
$ pip install openmim
$ pip install mmcv-full
$ mim install mmsegmentation
```

## <a name="file-structure"></a> 目录结构

```
SFNet
  |---- configs
  |       |---- _base_
  |       |       |---- datasets
  |       |       |        |---- cityscapes.py
  |       |       |---- models
  |       |       |        |---- sfnet_r18-d8_dsn.py
  |       |       |        |---- sfnet_r18-d8.py
  |       |       |        |---- sfnet_r50-d8.py
  |       |       |        |---- sfnet_r101-d8.py
  |       |       |---- schedules
  |       |       |        |---- schedule_50k.py
  |       |       |---- default_runtime.py
  |       |---- sfnet
  |               |---- sfnet_r18-d8_512x1024_50k_cityscapes.py
  |               |---- sfnet_r18-d8_dsn_512x1024_50k_cityscapes.py
  |               |---- sfnet_r50-d8_512x1024_50k_cityscapes.py
  |               |---- sfnet_r101-d8_512x1024_50k_cityscapes.py
  |---- mmseg
  |       |---- models
  |               |---- backbones
  |               |       |---- resnet.py
  |               |---- necks
  |                       |---- sf_neck.py
  |---- .gitignore
  |---- README_en-US.md
  |---- README.md
  |---- requirements.txt
  |---- setup.cfg
```

## <a name="usage"></a> 用法

### <a name="train"></a> 训练

```shell
$ PYTHONPATH='.':$PYTHONPATH mim train mmseg configs/sfnet/sfnet_r18-d8_512x1024_50k_cityscapes.py \
    --work-dir ./work_dirs/sfnet_r18-d8_512x1024_50k_cityscapes/ \
    --gpus 1
```

### <a name="test"></a> 测试

由于 Cityscapes 官方不提供测试集 Ground Truth，所以首先需要为测试集中的图像生成分割结果：

```shell
$ PYTHONPATH='.':$PYTHONPATH mim test mmseg configs/sfnet/sfnet_r18-d8_512x1024_50k_cityscapes.py \
    --checkpoint ./work_dirs/sfnet_r18-d8_512x1024_50k_cityscapes/epoch_300.pth \
    --format-only \
    --eval-options "imgfile_prefix=./results"
```

然后将所有分割结果打包压缩：

```shell
$ zip -r results.zip results/
```

最后手动将该压缩包上传到 [Cityscapes 官网的提交入口](https://www.cityscapes-dataset.com/submit/)即可。

## <a name="results"></a> 结果

**比赛要求在 cityscapes 上达到超过 77% 的 mIoU，复现得到的三个模型均达到了超过 78% 的 mIoU，满足精度要求。**

整体结果：

模型 | aAcc | mAcc | mIoU | FLOPs | Params | 链接
:---: | :---: | :---: | :---: | :---: | :---: | :---:
SFNet(ResNet-18) | 96.0 | 84.47 | **78.03** | 243.32G | 12.87M | [sfnet-resnet18-dsn.pth](https://drive.google.com/file/d/1nI1hzlgGZEGRGAZw-sryI6jLFatfuzw4/view?usp=sharing)
SFNet(ResNet-18) +dsn | 95.97 | 84.89 | **78.31** | 243.32G | 13.32M | [sfnet-resnet18.pth](https://drive.google.com/file/d/1vPIWndVDQkBdK2bFaR472wzTRHWjBcpb/view?usp=sharing)
SFNet(ResNet-50) | 96.42 | 86.23 | **80.17** | 664.07G | 31.33M | [sfnet-resnet50.pth](https://drive.google.com/file/d/1gXJsKxtIohWfPASNhQ7gzCRsxk-oRH3U/view?usp=sharing)
SFNet(ResNet-101) | 96.47 | 87.38 | **81.01** | 819.46G | 50.32M | [sfnet-resnet101.pth](https://drive.google.com/file/d/1kX4VxGUqFAn41w3EC1LnSIqWqdEcj9dt/view?usp=sharing)

在*验证集*各个类别上的 mIoU：

类别 | SFNet(ResNet-18) | SFNet(ResNet-50) | SFNet(ResNet-101)
:---: | :---: | :---: | :---:
road | 97.94 | 98.29 | 98.18
sidewalk | 83.78 | 85.88 | 85.39
building | 92.29 | 93.08 | 93.42
wall | 58.26 | 60.17 | 61.86
fence | 61.92 | 63.84 | 65.67
pole | 62.6 | 67.43 | 68.5
traffic light | 70.05 | 73.13 | 73.6
traffic sign | 77.61 | 80.73 | 81.4
vegetation | 92.19 | 92.86 | 92.85
terrain | 63.74 | 65.15 | 65.55
sky | 94.52 | 95.1 | 95.12
person | 81.75 | 83.03 | 84.36
rider | 62.46 | 65.28 | 69.83
car | 94.97 | 95.58 | 95.77
truck | 83.09 | 81.0 | 85.65
bus | 88.65 | 91.15 | 90.69
train | 81.89 | 84.39 | 84.81
motorcycle | 62.8 | 67.72 | 67.23
bicycle | 77.32 | 79.47 | 79.31

## <a name="citation"></a> 引用

```
@misc{mmcv,
    title={{MMCV: OpenMMLab} Computer Vision Foundation},
    author={MMCV Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmcv}},
    year={2018}
}
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```

