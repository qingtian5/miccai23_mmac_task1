model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer',
        arch='base',
        img_size=384,
        stage_cfgs=dict(block_cfgs=dict(window_size=12))),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='ArcFaceClsHead',
        num_classes=5,
        in_channels=1024,
        num_subcenters=3,
        loss = dict(type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=None))
dataset_type = 'ImageNet'
data_preprocessor = dict(
    num_classes=5,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=384,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=384, backend='pillow', interpolation='bicubic'),
    dict(type='PackInputs'),
]
train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        type='ImageNet',
        data_root=
        '../data/classification_train_val/Images/',
        ann_file=
        '../data/classification_enhanced_train/meta/train.txt',
        data_prefix='train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='RandomResizedCrop',
                scale=384,
                backend='pillow',
                interpolation='bicubic'),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackInputs'),
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        type='ImageNet',
        data_root=
        '../data/classification_val/Images/',
        ann_file=
        '../data/classification_val/meta/val.txt',
        data_prefix='val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Resize',
                scale=384,
                backend='pillow',
                interpolation='bicubic'),
            dict(type='PackInputs'),
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = dict(
    type='Accuracy', topk=(
        1,
        5,
    ))
test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    dataset=dict(
        type='ImageNet',
        data_root=
        '../data/classification_val/Images/',
        ann_file=
        '../data/classification_val/meta/val.txt',
        data_prefix='val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='Resize',
                scale=384,
                backend='pillow',
                interpolation='bicubic'),
            dict(type='PackInputs'),
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
test_evaluator = dict(
    type='Accuracy', topk=(
        1,
        5,
    ))
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.001/8,
        weight_decay=0.05,
        eps=1e-08,
        betas=(
            0.9,
            0.999,
        )),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys=dict({
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        })),
    clip_grad=dict(max_norm=5.0))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001/8,
        by_epoch=True,
        end=5,
        convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', eta_min=1e-05, by_epoch=True, begin=5),
]
train_cfg = dict(by_epoch=True, max_epochs=36, val_interval=3)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=32)
default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
        # dict(type='WandbVisBackend'),
    ])
log_level = 'INFO'
load_from = "../pretrained_ckpt/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth"
resume = False
randomness = dict(seed=None, deterministic=False)
work_dir = "../projects/submission/working"
