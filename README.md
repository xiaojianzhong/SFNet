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

