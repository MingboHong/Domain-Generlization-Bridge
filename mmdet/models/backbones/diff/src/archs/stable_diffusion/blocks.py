import torch
import torch.nn as nn
from typing import List
from .cim_utils import MultiScaleBasisBlock
import torch.nn.init as init
class LightAdapter(nn.Module):
    r"""
    See [`T2IAdapter`] for more information.
    """

    def __init__(
        self,
        in_channels: int = 4,
        channels: List[int] = [320, 640, 1280],
        num_res_blocks: int = 4,
        is_fdblock: bool = False,
        basis_reduction=2,
        basis_reduction_mode="sub",
        with_ep = True,
        with_query = True,
        with_mssp = False,
    ):
        super().__init__()

        in_channels = in_channels #* downscale_factor**2

       # self.unshuffle = nn.PixelUnshuffle(downscale_factor)   

        self.body = nn.ModuleList(
            [
                LightAdapterBlock(in_channels, channels[0], num_res_blocks,is_fdblock=is_fdblock,basis_reduction=basis_reduction, basis_reduction_mode=basis_reduction_mode, 
                                  with_ep = with_ep, with_query = with_query, with_mssp = with_mssp),
                *[
                    LightAdapterBlock(channels[i], channels[i + 1], num_res_blocks, down=True,is_fdblock=is_fdblock,basis_reduction=basis_reduction,
                                       basis_reduction_mode=basis_reduction_mode, with_ep = with_ep, with_query = with_query, with_mssp = with_mssp)
                    for i in range(len(channels) - 1)
                ],
                LightAdapterBlock(channels[-1], channels[-1], num_res_blocks, down=True,is_fdblock=is_fdblock,basis_reduction=basis_reduction, 
                                  basis_reduction_mode=basis_reduction_mode, with_ep = with_ep, with_query = with_query, with_mssp = with_mssp),
            ]
        )
        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #init.constant_(m.weight, 0)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""
        This method takes the input tensor x and performs downscaling and appends it in list of feature tensors. Each
        feature tensor corresponds to a different level of processing within the LightAdapter.
        """
       # x = self.unshuffle(x)

        features = []

        for block in self.body:
            x = block(x)
            features.append(x)

        return features


class LightAdapterBlock(nn.Module):
    r"""
    A `LightAdapterBlock` is a helper model that contains multiple `LightAdapterResnetBlocks`. It is used in the
    `LightAdapter` model.

    Parameters:
        in_channels (`int`):
            Number of channels of LightAdapterBlock's input.
        out_channels (`int`):
            Number of channels of LightAdapterBlock's output.
        num_res_blocks (`int`):
            Number of LightAdapterResnetBlocks in the LightAdapterBlock.
        down (`bool`, *optional*, defaults to `False`):
            Whether to perform downsampling on LightAdapterBlock's input.
    """

    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int, down: bool = False, is_fdblock=True,
                 basis_reduction=2, basis_reduction_mode="sub", with_ep = True, with_query = True, with_mssp =False):
        super().__init__()
        mid_channels = out_channels // 4
        self.is_fdblock = is_fdblock
        self.downsample = None
        if down:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.in_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.resnets = nn.Sequential(*[LightAdapterResnetBlock(mid_channels) for _ in range(num_res_blocks)])
        if self.is_fdblock:
            self.fdblocks = MultiScaleBasisBlock(in_channels=mid_channels, norm_cfg=dict(type='BN'),basis_reduction=basis_reduction,
                                                  basis_reduction_mode=basis_reduction_mode, with_ep = with_ep, with_query = with_query, with_mssp = with_mssp)
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        This method takes tensor x as input and performs downsampling if required. Then it applies in convolution
        layer, a sequence of residual blocks, and out convolutional layer.
        """
        if self.downsample is not None:
            x = self.downsample(x)

        x = self.in_conv(x)
        x = self.resnets(x)
        if self.is_fdblock:
            x = self.fdblocks(x)
        x = self.out_conv(x)

        return x


class LightAdapterResnetBlock(nn.Module):
    """
    A `LightAdapterResnetBlock` is a helper model that implements a ResNet-like block with a slightly different
    architecture than `AdapterResnetBlock`.

    Parameters:
        channels (`int`):
            Number of channels of LightAdapterResnetBlock's input and output.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.block2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        This function takes input tensor x and processes it through one convolutional layer, ReLU activation, and
        another convolutional layer and adds it to input tensor.
        """

        h = self.act(self.block1(x))
        h = self.block2(h)

        return h + x


# def test_light_adapter():
#     model = LightAdapter(
#         in_channels=4,
#         channels=[320, 640, 1280],
#         num_res_blocks=2,
#         downscale_factor=8
#     )
#     model.eval()

#     # 输入尺寸为 (C, H, W)，batch_size 不用于 FLOPs 计算
#     with torch.cuda.device(0):  # 或使用 CPU 删除该行
#         macs, params = get_model_complexity_info(model, (4, 128, 128), as_strings=True,
#                                                  print_per_layer_stat=False, verbose=False)
#         print(f"FLOPs (MACs): {macs}")
#         print(f"Parameters: {params}")

# test_light_adapter()