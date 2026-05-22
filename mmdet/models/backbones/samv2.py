import torch
import torch.nn as nn


from mmdet.registry import MODELS
import torch.nn.functional as F

from mmdet.models.backbones.sam.hieradet import Hiera
from mmdet.models.backbones.sam.image_encoder import ImageEncoder, FpnNeck
from mmengine.model import BaseModule
from .cim_utils import MultiScaleBasisBlock

@MODELS.register_module()
class SAM(BaseModule):
    def __init__(self,  is_fdblock=False, out_channels=[144, 288, 576, 1152],
                  init_cfg=None):
        super(SAM,self).__init__(init_cfg)
        # backbone
        self.is_fdblock = is_fdblock
        self.out_channels = out_channels
         # image encoder
        self.image_encoder = ImageEncoder(
            trunk=Hiera(),
            neck=FpnNeck(),
        )
        self.is_fdblock = is_fdblock
        if self.is_fdblock:
            self.fdblocks =  nn.ModuleList()
            for ch in self.out_channels:
                self.fdblocks.append(MultiScaleBasisBlock(num_basis=ch, in_channels=ch))
        for name, param in self.named_parameters():
            if "fdblocks" not in name:
                param.requires_grad = False
            else:
                print(f"{name} is trainable")
    def forward(self, input: torch.Tensor):
        features = self.image_encoder(input)
        backbone_fpn = features["backbone_feats"]

        out = []
        if not self.is_fdblock:
            return backbone_fpn
        for x, block in zip(backbone_fpn, self.fdblocks):
            x = block(x)
            out.append(x)
        return out

