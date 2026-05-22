ns_type = 'CocoDataset'
ns_root = 'data/'



ns_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale_factor=1.0, keep_ratio=True),
     #dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
test_ns = dict(
    type=ns_type,
    data_root=ns_root,
    metainfo=dict(classes=('bus','bike','car','motor','person','rider','truck')),
    ann_file='night_sunny.json',
    data_prefix=dict(img='diverseWeather/night_sunny/VOC2007/JPEGImages'),
    test_mode=True,
    pipeline=ns_test_pipeline,
    filter_cfg=dict(filter_empty_gt=True),
    backend_args=None
)