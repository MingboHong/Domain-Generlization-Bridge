df_type = 'CocoDataset'
df_root = 'data/'


df_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale_factor=1.0, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_df = dict(
    type=df_type,
    data_root=df_root,
    metainfo=dict(classes=('bus','bike','car','motor','person','rider','truck')),
    ann_file='daytime_foggy.json',
    data_prefix=dict(img='diverseWeather/daytime_foggy/VOC2007/JPEGImages'),
    test_mode=True,
    pipeline=df_test_pipeline,
    filter_cfg=dict(filter_empty_gt=True),
    backend_args=None
)