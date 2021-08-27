# model settings
custom_imports = dict(imports=[
    'mmseg.models.backbones.resnet',
    'mmseg.models.necks.sf_neck',
], allow_failed_imports=False)

norm_cfg = dict(type='SyncBN', requires_grad=True)
sampler = dict(
    type='OHEMPixelSampler',
    thresh=0.7,
    min_kept=10000)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(
        type='ResNetV1c',
        stem_channels=128,
        depth=101,
        dilations=(1, 1, 2, 4),
        norm_cfg=norm_cfg),
    neck=dict(
        type='SFNeck',
        in_channels=[256, 512, 1024, 2048],
        channels=256,
        align_corners=True,
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=4,
        channels=256,
        num_convs=1,
        kernel_size=3,
        concat_input=False,
        num_classes=19,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        sampler=sampler,
        align_corners=True),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
