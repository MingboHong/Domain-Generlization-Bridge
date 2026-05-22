edark_type = 'CocoDataset'
edark_root =  'data/DroneVehicle/'


edark_test_pipeline = [
   dict(type='LoadImageFromFile', backend_args=None),
     dict(type='Resize', scale=(840, 712), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_edark = dict(
    type=edark_type,
    data_root=edark_root,
    metainfo=dict(classes=('car', 'freight_car', 'truck', 'bus', 'van')),
    ann_file='extreme_dark.json',
    data_prefix=dict(img='extreme/'),
    test_mode=True,
    pipeline=edark_test_pipeline,
    filter_cfg=dict(filter_empty_gt=True),
    backend_args=None
)