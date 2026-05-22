cp_type = 'VOCDGDataset'
cp_root = 'data/clipart'




cp_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1200, 600), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_cp = dict(
    type=cp_type,
    data_root=cp_root,
    ann_file='VOC2007/ImageSets/Main/train_test.txt',
    data_prefix=dict(sub_data_root='VOC2007/'),
    test_mode=True,
    pipeline=cp_test_pipeline,
    filter_cfg=dict(filter_empty_gt=True),
    backend_args=None
)