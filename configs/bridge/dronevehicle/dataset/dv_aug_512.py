_base_ = [
    "../eval/foggy_512.py",
    "../eval/dark_512.py",
    "../eval/edark_512.py",

]


# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/DroneVehicle'
classes = ('car', 'freight_car', 'truck', 'bus', 'van')

backend_args = None

color_space = [
    [dict(type='ColorTransform')],
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),

    dict(
        type='RandomResize',
        scale=(640, 512),
        ratio_range=(0.5, 1.5),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(512, 512),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='Pad',size=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(type='AlbuDomainAdaption', domain_adaption_type='ALL',
         target_dir='data/DroneVehicle/day', p=0.5, is_city=True),
    dict(type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='day.json',
        data_prefix=dict(img='day/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale_factor=1.0, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
        {{_base_.test_dark}},
        {{_base_.test_edark}},
        {{_base_.test_foggy}},
 
        ]
    ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='DGCocoMetric',
    dataset_keys=[
        "dark",
        "extreme",
        "foggy"
        ],
     
    ann_files={
        "dark": "data/DroneVehicle/dark.json",
        "extreme": "data/DroneVehicle/extreme_dark.json",
        "foggy": "data/DroneVehicle/foggy.json"
    },
    classwise = True,
    metric='bbox',
    format_only=False,
    outfile_prefix='results/dv/baseline',
    )
test_evaluator = val_evaluator
