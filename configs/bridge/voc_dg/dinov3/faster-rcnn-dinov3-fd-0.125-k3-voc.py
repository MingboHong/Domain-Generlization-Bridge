_base_ = ['./faster-rcnn-dinov3-fd-0.125-voc.py']

model = dict(
    backbone=dict(conv_kernel_size=3),
    roi_head=dict(conv_kernel_size=3))
