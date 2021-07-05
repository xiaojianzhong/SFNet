# Semantic Flow for Fast and Accurate Scene Parsing

基于 [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 的 SFNet 复现。

原始论文：[Semantic Flow for Fast and Accurate Scene Parsing](https://arxiv.org/abs/2002.10120)

阅读其他语言的版本：[English](./README.md) | 简体中文

## 目录

- [环境要求](#prerequisites)
- [目录结构](#file-structure)
- [用法](#usage)
- [结果](#results)

## <a name="prerequisites"></a> 环境要求

- [mim](https://github.com/open-mmlab/mim) >= v0.1.0
- [mmcv-full](https://github.com/open-mmlab/mmcv) >= v1.3.8
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) >= v0.15.0

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
  |       |       |        |---- sfnet_r50-d8.py
  |       |       |---- schedules
  |       |       |        |---- schedule_50k.py
  |       |       |---- default_runtime.py
  |       |---- sfnet
  |               |---- sfnet_r18-d8_512x1024_50k_cityscapes.py
  |---- sf_neck.py
```

## <a name="usage"></a> 用法

```shell
PYTHONPATH='.':$PYTHONPATH mim train mmseg configs/sfnet/sfnet_r18-d8_512x1024_50k_cityscapes.py \
    --work-dir ./work_dirs/sfnet_r18-d8_512x1024_50k_cityscapes/ \
    --gpus 1
```

## <a name="results"></a> 结果

