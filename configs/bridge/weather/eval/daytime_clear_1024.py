dc_type = 'CocoDataset'
dc_root = 'data/'



dc_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
      dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
      dict(type='Pad',size=(1024, 1024)),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_dc = dict(
    type=dc_type,
    data_root=dc_root,
    metainfo=dict(classes=('bus','bike','car','motor','person','rider','truck')),
    ann_file='daytime_clear_test.json',
    data_prefix=dict(img='diverseWeather/daytime_clear/VOC2007/JPEGImages'),
    test_mode=True,
    pipeline=dc_test_pipeline,
    filter_cfg=dict(filter_empty_gt=True),
    backend_args=None
)

