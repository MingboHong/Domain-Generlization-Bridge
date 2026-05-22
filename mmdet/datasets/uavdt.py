from mmdet.registry import DATASETS
from .coco import CocoDataset,COCO
import os.path as osp
from typing import List, Union
@DATASETS.register_module()
class UAVDataset(CocoDataset):
    """Dataset for UAVDT."""

    METAINFO = {
        'classes':
        ('car'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142)]
    }
    


