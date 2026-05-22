nr_type = 'CocoDataset'
nr_root = 'data/'



nr_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale_factor=1.0, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


test_nr = dict(
    type=nr_type,
    data_root=nr_root,
    metainfo=dict(classes=('bus','bike','car','motor','person','rider','truck')),
    ann_file='night_rainy.json',
    data_prefix=dict(img='diverseWeather/night_rainy/VOC2007/JPEGImages'),
    test_mode=True,
    pipeline=nr_test_pipeline,
    filter_cfg=dict(filter_empty_gt=True),
    backend_args=None
)