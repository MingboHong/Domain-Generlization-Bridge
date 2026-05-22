clear_type = 'CocoDataset'
clear_root = 'data/cityscapes_clear'


clear_test_pipeline = [
   dict(type='LoadImageFromFile', backend_args=None),
     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_clear = dict(
    type=clear_type,
    data_root=clear_root,
    metainfo=dict(classes=('bicycle', 'bus', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck')),
   ann_file='cityscapes_val.json',
    data_prefix=dict(img='leftImg8bit/val'),
    test_mode=True,
    pipeline=clear_test_pipeline,
    filter_cfg=dict(filter_empty_gt=True),
    backend_args=None
)