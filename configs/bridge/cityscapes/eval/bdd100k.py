bdd_type = 'CocoDataset'
bdd_root = 'data/BDD100K'


bdd_test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


test_bdd = dict(
    type=bdd_type,
    data_root=bdd_root,
    metainfo=dict(classes=('bicycle', 'bus', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck')),
    ann_file='bdd100k_val.json',
    data_prefix=dict(img='val/'),
    test_mode=True,
    filter_cfg=dict(filter_empty_gt=True),
    pipeline=bdd_test_pipeline,
    backend_args=None
)