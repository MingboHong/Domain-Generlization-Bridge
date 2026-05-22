_base_ = ['./faster-rcnn-dinov3-fd-0.125-voc.py']

model = dict(
    backbone=dict(basis_normalize=False),
    roi_head=dict(basis_normalize=False))
