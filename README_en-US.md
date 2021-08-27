# Semantic Flow for Fast and Accurate Scene Parsing

SFNet implementation based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), which is submitted for [OpenMMLab Algorithm Ecological Challenge](https://openmmlab.com/competitions/algorithm-2021).

The original paper: [Semantic Flow for Fast and Accurate Scene Parsing](https://arxiv.org/abs/2002.10120)

Read this in other languages: English | [简体中文](./README.md)

## Table of Contents

- [Prerequisites](#prerequisites)
- [File Structure](#file-structure)
- [Usage](#usage)
    - [Train](#train)
    - [Test](#test)
- [Results](#results)
- [Citation](#citation)

## <a name="prerequisites"></a> Prerequisites

- [mim](https://github.com/open-mmlab/mim) == v0.1.0
- [mmcv-full](https://github.com/open-mmlab/mmcv) == v1.3.8
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) == v0.15.0

```shell
$ pip install openmim
$ pip install mmcv-full
$ mim install mmsegmentation
```

## <a name="file-structure"></a> File Structure

```
SFNet
  |---- configs
  |       |---- _base_
  |       |       |---- datasets
  |       |       |        |---- cityscapes.py
  |       |       |---- models
  |       |       |        |---- sfnet_r18-d8.py
  |       |       |---- schedules
  |       |       |        |---- schedule_50k.py
  |       |       |---- default_runtime.py
  |       |---- sfnet
  |               |---- sfnet_r18-d8_512x1024_50k_cityscapes.py
  |---- sf_neck.py
```

## <a name="usage"></a> Usage

### <a name="train"></a> Train

```shell
$ PYTHONPATH='.':$PYTHONPATH mim train mmseg configs/sfnet/sfnet_r18-d8_512x1024_50k_cityscapes.py \
    --work-dir ./work_dirs/sfnet_r18-d8_512x1024_50k_cityscapes/ \
    --gpus 1
```

### <a name="test"></a> Test

Due to the absence of the Ground Truth for the Cityscapes *test* set, we need to generate segmentation results for these images firstly:

```shell
$ PYTHONPATH='.':$PYTHONPATH mim test mmseg configs/sfnet/sfnet_r18-d8_512x1024_50k_cityscapes.py \
    --checkpoint ./work_dirs/sfnet_r18-d8_512x1024_50k_cityscapes/epoch_300.pth \
    --format-only \
    --eval-options "imgfile_prefix=./results"
```

compress all these segmentation results into one file:

```shell
$ zip -r results.zip results/
```

and finally upload the .zip file onto [the submission entry for the Cityscapes dataset](https://www.cityscapes-dataset.com/submit/).

## <a name="results"></a> Results

**The contest requires the metric mIoU to be larger than 77%, while all the three models of our reimplementation achieve more than 78%, which meets the accuracy requirement.**

Overall results:

model | aAcc | mAcc | mIoU | FLOPs | Params | link
:---: | :---: | :---: | :---: | :---: | :---: | :---:
SFNet(ResNet-18) | 96.0 | 84.47 | **78.03** | 243.32G | 12.87M | [sfnet-resnet18-dsn.pth](https://drive.google.com/file/d/1nI1hzlgGZEGRGAZw-sryI6jLFatfuzw4/view?usp=sharing)
SFNet(ResNet-18) +dsn | 95.97 | 84.89 | **78.31** | 243.32G | 13.32M | [sfnet-resnet18.pth](https://drive.google.com/file/d/1vPIWndVDQkBdK2bFaR472wzTRHWjBcpb/view?usp=sharing)
SFNet(ResNet-50) | 96.42 | 86.23 | **80.17** | 664.07G | 31.33M | [sfnet-resnet50.pth](https://drive.google.com/file/d/1gXJsKxtIohWfPASNhQ7gzCRsxk-oRH3U/view?usp=sharing)
SFNet(ResNet-101) | 96.47 | 87.38 | **81.01** | 819.46G | 50.32M | [sfnet-resnet101.pth](https://drive.google.com/file/d/1kX4VxGUqFAn41w3EC1LnSIqWqdEcj9dt/view?usp=sharing)

mIoU on each class of the *val* set:

class | SFNet(ResNet-18) | SFNet(ResNet-50) | SFNet(ResNet-101)
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

## <a name="citation"></a> Citation

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
