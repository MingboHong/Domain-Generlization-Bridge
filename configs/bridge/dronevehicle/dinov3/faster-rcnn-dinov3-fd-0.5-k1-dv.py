_base_ = ['./faster-rcnn-dinov3-fd-0.5-dv.py']

model = dict(
    backbone=dict(conv_kernel_size=1),
    roi_head=dict(conv_kernel_size=1))
