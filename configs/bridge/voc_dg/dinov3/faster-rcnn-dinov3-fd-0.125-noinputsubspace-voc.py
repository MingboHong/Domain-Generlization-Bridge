_base_ = ['./faster-rcnn-dinov3-fd-0.125-voc.py']

model = dict(
    backbone=dict(with_input_subspace=False),
    roi_head=dict(with_input_subspace=False))
