dark_type = 'CocoDataset'
dark_root =  'data/DroneVehicle/'


dark_test_pipeline = [
   dict(type='LoadImageFromFile', backend_args=None),
     dict(type='Resize', scale=(840, 712), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_dark = dict(
    type=dark_type,
    data_root=dark_root,
    metainfo=dict(classes=('car', 'freight_car', 'truck', 'bus', 'van')),
    ann_file='dark.json',
    data_prefix=dict(img='dark/'),
    test_mode=True,
    pipeline=dark_test_pipeline,
    filter_cfg=dict(filter_empty_gt=True),
    backend_args=None
)