dr_type = 'CocoDataset'
dr_root = 'data/'


dr_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None ),
    dict(type='Resize', scale_factor=1.0, keep_ratio=True),
    # dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_dr = dict(
    type=dr_type,
    data_root=dr_root,
    metainfo=dict(classes=('bus','bike','car','motor','person','rider','truck')),
    ann_file='dusk_rainy.json',
    data_prefix=dict(img='diverseWeather/dusk_rainy/VOC2007/JPEGImages'),
    test_mode=True,
    pipeline=dr_test_pipeline,
    filter_cfg=dict(filter_empty_gt=True),
    backend_args=None
)