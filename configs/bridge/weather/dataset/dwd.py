_base_ = [
   "../eval/night_sunny.py",
     "../eval/dusk_rainy.py",
    "../eval/night_rainy.py",
    "../eval/daytime_foggy.py",
]


# dataset settings
dc_type = 'CocoDataset'
dc_root = 'data'
classes = ('bus','bike','car','motor','person','rider','truck')

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

geometric = [
    [dict(type='Rotate')],
    [dict(type='ShearX')],
    [dict(type='ShearY')],
    [dict(type='TranslateX')],
    [dict(type='TranslateY')],
]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=[
         (2048, 1024), (2048, 800)], keep_ratio=True),
    dict(type='RandomCrop', crop_type='absolute', crop_size=(1024, 1024),
         recompute_bbox=True, allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(type='AlbuDomainAdaption', domain_adaption_type='ALL',
         target_dir='data/diverseWeather/daytime_clear/VOC2007/ImageSets/Main/train.txt', p=0.5,is_weather=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction',
                   'homography_matrix')),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
     dict(type='Resize', scale_factor=1.0, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dc_type,
        data_root=dc_root,
        metainfo=dict(classes=classes),
        ann_file='daytime_clear_train.json',
        data_prefix=dict(img='diverseWeather/daytime_clear/VOC2007/JPEGImages'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.test_ns}},
            {{_base_.test_dr}},
            {{_base_.test_nr}},
          {{_base_.test_df}},
        ]
    ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='DGCocoMetric',
    dataset_keys=[
        "night_sunny",
        "dusk_rainy",
        "night_rainy",
       "daytime_foggy"
        ],
    ann_files={
        "night_sunny": "data/night_sunny.json",
        "dusk_rainy": "data/dusk_rainy.json",
        "night_rainy": "data/night_rainy.json",
        "daytime_foggy": "data/daytime_foggy.json"
    },
    classwise = True,
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator
