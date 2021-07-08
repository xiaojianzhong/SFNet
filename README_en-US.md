# Semantic Flow for Fast and Accurate Scene Parsing

SFNet implementation based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

The original paper: [Semantic Flow for Fast and Accurate Scene Parsing](https://arxiv.org/abs/2002.10120)

Read this in other languages: English | [简体中文](./README.md)

## Table of Contents

- [Prerequisites](#prerequisites)
- [File Structure](#file-structure)
- [Usage](#usage)
- [Results](#results)

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

```shell
$ PYTHONPATH='.':$PYTHONPATH mim train mmseg configs/sfnet/sfnet_r18-d8_512x1024_50k_cityscapes.py \
    --work-dir ./work_dirs/sfnet_r18-d8_512x1024_50k_cityscapes/ \
    --gpus 1
```

## <a name="results"></a> Results
