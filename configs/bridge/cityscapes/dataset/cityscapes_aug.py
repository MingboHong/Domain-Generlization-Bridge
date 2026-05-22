_base_ = [
    "../eval/bdd100k.py",
    "../eval/cityscapes_foggy.py",
    "../eval/cityscapes_clear.py",

]


# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/'
classes = ('bicycle', 'bus', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck')

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
        scale=(1333, 800),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(800, 1333),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(type='AlbuDomainAdaption', domain_adaption_type='ALL',
         
         target_dir='data/cityscapes_clear/leftImg8bit/train', p=0.5, is_city=True),
    dict(type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='cityscapes_clear/cityscapes_train.json',
        data_prefix=dict(img='cityscapes_clear/leftImg8bit/train'),
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
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
        {{_base_.test_clear}},
        {{_base_.test_bdd}},
        {{_base_.test_foggy}},
 
        ]
    ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='DGCocoMetric',
    dataset_keys=[
        "cityscapes_clear",
        "BDD100K",
        "cityscapes_foggy"
        ],
    ann_files={
       "cityscapes_clear": "data/cityscapes_clear/cityscapes_val.json",
        "BDD100K": "data/BDD100K/bdd100k_val.json",
        "cityscapes_foggy": "data/cityscapes_foggy/foggy_cityscapes.json"
    },
    classwise = True,
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator
