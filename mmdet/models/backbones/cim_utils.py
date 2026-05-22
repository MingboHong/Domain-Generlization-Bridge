import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


def _build_orthogonal_basis(num_basis, feat_dim):
    basis = torch.randn(num_basis, feat_dim)
    q, _ = torch.linalg.qr(basis.T)
    return nn.Parameter(q.T)


def _get_same_padding(kernel_size):
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f'kernel_size must be a positive odd integer, got {kernel_size}')
    return kernel_size // 2


def _project_onto_basis(x, basis, eps=1e-5, normalize_basis=True):
    batch, channels, height, width = x.shape
    assert channels == basis.shape[1]

    basis = basis.to(device=x.device, dtype=x.dtype)
    if normalize_basis:
        basis = F.normalize(basis, p=2, dim=1)
    x_flat = x.permute(0, 2, 3, 1).reshape(-1, channels)
    gram = basis @ basis.T
    reg = eps * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    coeffs = torch.linalg.solve(gram + reg, (x_flat @ basis.T).T).T
    x_proj = coeffs @ basis
    return x_proj.view(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()


def _resolve_basis_dim(in_channels, basis_reduction, basis_reduction_mode):
    assert basis_reduction_mode in ['sub', 'div', 'mul']
    if basis_reduction_mode == 'sub':
        reduced = in_channels - basis_reduction
    elif basis_reduction_mode == 'div':
        reduced = in_channels // basis_reduction
    else:
        reduced = int(in_channels * basis_reduction)
    reduced = int(reduced)
    if reduced <= 0 or reduced > in_channels:
        raise ValueError(
            f'Invalid reduced basis dimension {reduced} for '
            f'in_channels={in_channels}, basis_reduction={basis_reduction}, '
            f'mode={basis_reduction_mode}')
    return reduced


class ExpectationEstimator(nn.Module):
    def __init__(self,
                 feat_dim=256,
                 num_basis=128,
                 num_samples=128,
                 eps=1e-5,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_query=True,
                 with_ssp=True,
                 basis_normalize=True,
                 conv_kernel_size=3):
        super().__init__()
        self.eps = eps
        self.with_query = with_query
        self.with_ssp = with_ssp
        self.basis_normalize = basis_normalize
        conv_padding = _get_same_padding(conv_kernel_size)

        self.sample_query_proj = ConvModule(
            feat_dim,
            num_samples,
            kernel_size=1,
            stride=1,
            norm_cfg=None,
            act_cfg=None)
        self.expectation_basis = _build_orthogonal_basis(num_basis, feat_dim)
        self.refine_conv = ConvModule(
            feat_dim,
            feat_dim,
            kernel_size=conv_kernel_size,
            stride=1,
            padding=conv_padding,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        if self.with_query:
            sample_response_map = self.sample_query_proj(x)
            sample_attention = F.softmax(sample_response_map, dim=1)
            expected_feature_map = (
                sample_response_map * sample_attention).sum(dim=1, keepdim=True) * x
        else:
            expected_feature_map = x

        if self.with_ssp:
            expected_feature_map = _project_onto_basis(
                expected_feature_map,
                self.expectation_basis,
                eps=self.eps,
                normalize_basis=self.basis_normalize)

        return self.refine_conv(expected_feature_map)


class MultiScaleBasisBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 basis_reduction=2,
                 basis_reduction_mode='div',
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='SiLU'),
                 with_ssp=True,
                 with_query=True,
                 with_input_subspace=False,
                 with_dropout=False,
                 basis_normalize=True,
                  conv_kernel_size=3):
        super().__init__()
        self.with_input_subspace = with_input_subspace
        self.with_dropout = with_dropout
        self.basis_normalize = basis_normalize
        self.num_reduced_basis = _resolve_basis_dim(
            in_channels, basis_reduction, basis_reduction_mode)
        conv_padding = _get_same_padding(conv_kernel_size)

        if self.with_input_subspace:
            self.input_basis = _build_orthogonal_basis(
                self.num_reduced_basis, in_channels)

        self.mediator_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size=conv_kernel_size,
            stride=1,
            padding=conv_padding,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.expected_input_estimator = ExpectationEstimator(
            feat_dim=in_channels,
            num_basis=self.num_reduced_basis,
            num_samples=in_channels,
            with_ssp=with_ssp,
            with_query=with_query,
            basis_normalize=basis_normalize,
            conv_kernel_size=conv_kernel_size)
        self.expected_mediator_estimator = ExpectationEstimator(
            feat_dim=in_channels,
            num_basis=self.num_reduced_basis,
            num_samples=in_channels,
            with_ssp=with_ssp,
            with_query=with_query,
            basis_normalize=basis_normalize,
            conv_kernel_size=conv_kernel_size)
        self.fusion_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size=conv_kernel_size,
            stride=1,
            padding=conv_padding,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, ConvModule):
                conv = module.conv
                if isinstance(conv, nn.Conv2d):
                    nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
                    if conv.bias is not None:
                        nn.init.zeros_(conv.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        mediator_feature = self.mediator_conv(x)
        expected_input_feature = self.expected_input_estimator(x)
        expected_mediator_feature = self.expected_mediator_estimator(
            mediator_feature)
        if self.with_input_subspace:
            mediator_feature = _project_onto_basis(
                x,
                self.input_basis,
                normalize_basis=self.basis_normalize)

        fused_feature = (
            mediator_feature +
            expected_input_feature +
            expected_mediator_feature)
        return self.fusion_conv(fused_feature)
