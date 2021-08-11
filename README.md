# Semantic Flow for Fast and Accurate Scene Parsing

基于 [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 的 SFNet 复现。

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
  |       |       |        |---- sfnet_r18-d8.py
  |       |       |---- schedules
  |       |       |        |---- schedule_50k.py
  |       |       |---- default_runtime.py
  |       |---- sfnet
  |               |---- sfnet_r18-d8_512x1024_50k_cityscapes.py
  |---- sf_neck.py
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

整体结果：

模型 | *val* aAcc | *val* mAcc | *val* mIoU | *test* mIoU | 链接
:---: | :---: | :---: | :---: | :---: | :---:
SFNet(ResNet-18) | 95.68 | 82.62 | 75.72 | 74.46 | [sfnet-resnet18.pth](https://drive.google.com/file/d/1vPIWndVDQkBdK2bFaR472wzTRHWjBcpb/view?usp=sharing)

在各个类别上的 mIoU：

类别 | SFNet(ResNet-18) *val* | SFNet(ResNet-18) *test*
:---: | :---: | :---:
road | 97.75 | 97.96
sidewalk | 82.47 | 81.66
building | 91.82 | 91.67
wall | 45.84 | 41.71
fence | 56.10 | 50.41
pole | 63.78 | 61.78
traffic light | 67.45 | 68.07
traffic sign | 77.5 | 72.89
vegetation | 91.92 | 92.64
terrain | 60.98 | 69.63
sky | 94.70 | 95.04
person | 80.24 | 83.26
rider | 58.23 | 65.34
car | 94.77 | 94.82
truck | 79.75 | 61.02
bus | 85.06 | 75.09
train | 72.14 | 79.04
motorcycle | 62.17 | 60.88
bicycle | 75.97 | 71.88

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

