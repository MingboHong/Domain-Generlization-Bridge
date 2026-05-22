_base_ = ['./faster-rcnn-dinov3-fd-0.125-voc.py']

model = dict(
    backbone=dict(with_query=True),
    roi_head=dict(with_query=True))
