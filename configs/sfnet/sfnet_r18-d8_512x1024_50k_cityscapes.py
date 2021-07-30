_base_ = [
    '../_base_/models/sfnet_r18-d8.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_50k.py'
]

lr_config = dict(policy='poly', power=1.0, min_lr=1e-4, by_epoch=True)
runner = dict(type='EpochBasedRunner', max_iters=300)
checkpoint_config = dict(by_epoch=True, interval=30)
evaluation = dict(interval=30, metric='mIoU')

# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 placeholder
fp16 = dict()
