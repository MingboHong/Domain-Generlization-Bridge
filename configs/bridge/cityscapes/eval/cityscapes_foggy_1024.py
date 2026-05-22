foggy_type = 'CocoDataset'
foggy_root = 'data/cityscapes_foggy'



foggy_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
      dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
      dict(type='Pad',size=(1024, 1024)),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


test_foggy = dict(
    type=foggy_type,
    data_root=foggy_root,
    metainfo=dict(classes=('bicycle', 'bus', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck')),
    ann_file='foggy_cityscapes.json',
    data_prefix=dict(img='leftImg8bit_foggy/val/'),
    test_mode=True,
    pipeline=foggy_test_pipeline,
    filter_cfg=dict(filter_empty_gt=True),
    backend_args=None
)