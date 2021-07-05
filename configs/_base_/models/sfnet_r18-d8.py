# model settings
custom_imports = dict(imports=['sf_neck'], allow_failed_imports=False)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=18,
        norm_cfg=norm_cfg,
        contract_dilation=True),
    neck=dict(
        type='SFNeck',
        in_channels=[64, 128, 256, 512],
        channels=128,
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=4,
        channels=128,
        num_convs=1,
        kernel_size=1,
        num_classes=19,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            in_index=1,
            channels=128,
            num_convs=1,
            num_classes=19,
            norm_cfg=norm_cfg,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=128,
            in_index=2,
            channels=128,
            num_convs=1,
            num_classes=19,
            norm_cfg=norm_cfg,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=128,
            in_index=3,
            channels=128,
            num_convs=1,
            num_classes=19,
            norm_cfg=norm_cfg,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
