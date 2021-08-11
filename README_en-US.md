# Semantic Flow for Fast and Accurate Scene Parsing

SFNet implementation based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

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
