foggy_type = 'CocoDataset'
foggy_root =  'data/DroneVehicle/'


foggy_test_pipeline = [
   dict(type='LoadImageFromFile', backend_args=None),
     dict(type='Resize', scale=(840, 712), keep_ratio=True),
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
    metainfo=dict(classes=('car', 'freight_car', 'truck', 'bus', 'van')),
    ann_file='foggy.json',
    data_prefix=dict(img='foggy/'),
    test_mode=True,
    pipeline=foggy_test_pipeline,
    filter_cfg=dict(filter_empty_gt=True),
    backend_args=None
)