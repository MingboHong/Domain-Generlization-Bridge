import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule



class ExpectationEstimator(nn.Module):
    def __init__(self, feat_dim=256, num_basis=128, num_samples=128, eps=1e-5,
                 norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'),with_query=True, with_ssp=True):
        
        # dict(type='GN', num_groups=32)   dict(type='SiLU')
        # norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')
        super().__init__()
        self.feat_dim = feat_dim
        self.num_basis = num_basis
        self.num_samples = num_samples
        self.eps = eps
        self.with_query = with_query
        self.with_ssp = with_ssp
        # 1x1 conv to generate token / query map
        self.q_conv = ConvModule(feat_dim, num_samples, kernel_size=1, stride=1,
                                 norm_cfg=None, act_cfg=None)

        # subspace basis for SSP
        basis = torch.randn(num_basis, feat_dim)  # (K, C)
        Q, _ = torch.linalg.qr(basis.T)
        orthogonal_basis = Q.T  # (K, C)
        self.register_parameter('basis', nn.Parameter(orthogonal_basis))

        # optional output conv
        self.out_conv = ConvModule(feat_dim, feat_dim, kernel_size=3, stride=1, padding=1,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)

    def ssp(self, x, V, eps=1e-5):
        B, C, H, W = x.shape
        assert C == V.shape[0]
        V = V.to(x.dtype).to(x.device)
        x_vec = x.permute(0, 2, 3, 1).reshape(-1, C)  # (N, d)
        
        temp = x_vec @ V  # (N, K)
        VtV = V.T @ V
        K = VtV.shape[0]
        reg = eps * torch.eye(K, device=VtV.device, dtype=VtV.dtype)
        alpha_T = torch.linalg.solve(VtV + reg, temp.T)  # (K, N)
        alpha = alpha_T.T  # (N, K)
        x_proj = alpha @ V.T  # (N, d)
        x_out = x_proj.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x_out

    def forward(self, x):
        # generate token / query
        if self.with_query:
            samples_feats = self.q_conv(x)  # (B, S, H, W)

            # spatial softmax + weighted sum -> expectation map
            weights = (samples_feats*F.softmax(samples_feats, dim=1)).sum(dim=1, keepdim=True)  # (B, S, H, W)
            e_map = weights * x  # (B, C, H, W)
        else:
            e_map = x
        if self.with_ssp:
            # subspace projection
            x_proj = self.ssp(e_map, self.basis.T)
        else:
            x_proj = e_map

        # optional out_conv
        out = self.out_conv(x_proj)
        return out
    


class MultiScaleBasisBlock(nn.Module):
    def __init__(self, in_channels, basis_reduction=2, basis_reduction_mode='div',
                 norm_cfg=dict(type='GN', num_groups=32), act_cfg = dict(type='SiLU'),with_ep=True,with_ssp=True,with_query=True, with_mssp= False):
        super().__init__()
        assert basis_reduction_mode in ['sub', 'div', 'mul']
        self.input_dim = in_channels

        self.with_ssp = with_ssp
        self.with_query = with_query
        if basis_reduction_mode == 'sub':
            self.re_basis = in_channels - basis_reduction
        elif basis_reduction_mode == 'div':
            self.re_basis = in_channels // basis_reduction
        elif basis_reduction_mode == 'mul':
            self.re_basis = int(in_channels * basis_reduction)
        self.with_ep = with_ep
        self.with_mssp = with_mssp

        if self.with_mssp:
            basis = torch.randn(self.re_basis, in_channels)  # (K, C)
            Q, _ = torch.linalg.qr(basis.T)
            orthogonal_basis = Q.T  # (K, C)
            self.register_parameter('x_basis', nn.Parameter(orthogonal_basis))
   
        self.m_proj = ConvModule(in_channels, in_channels, 3, stride=1, padding=1,
                                 norm_cfg=norm_cfg, act_cfg=act_cfg)

        if with_ep:
            self.ex = ExpectationEstimator(feat_dim=in_channels,
                                                    num_basis=self.re_basis,
                                                    num_samples=in_channels,
                                                    with_ssp=self.with_ssp,
                                                    with_query=self.with_query)
    
            self.em = ExpectationEstimator(feat_dim=in_channels,
                                                    num_basis=self.re_basis,
                                                    num_samples=in_channels,
                                                    with_ssp=self.with_ssp,
                                                    with_query=self.with_query)
  
     
            self.out_conv = ConvModule(in_channels, in_channels, 3, stride=1, padding=1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.init_weights() 

    def ssp(self, x, V, eps=1e-5):
        B, C, H, W = x.shape
        assert C == V.shape[0]
        V = V.to(x.dtype).to(x.device)
        x_vec = x.permute(0, 2, 3, 1).reshape(-1, C)  # (N, d)
        
        temp = x_vec @ V  # (N, K)
        VtV = V.T @ V
        K = VtV.shape[0]
        reg = eps * torch.eye(K, device=VtV.device, dtype=VtV.dtype)
        alpha_T = torch.linalg.solve(VtV + reg, temp.T)  # (K, N)
        alpha = alpha_T.T  # (N, K)
        x_proj = alpha @ V.T  # (N, d)
        x_out = x_proj.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x_out



    def init_weights(self):
        for m in self.modules():  
            if isinstance(m, ConvModule):
                conv = m.conv
                if isinstance(conv, nn.Conv2d):
                    nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
                    if conv.bias is not None:
                        nn.init.zeros_(conv.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
   
        m = self.m_proj(x)
    
        if self.with_ep:
            ex = self.ex(x)
            em = self.em(m)
            if self.with_mssp:
                m = self.ssp(x, self.x_basis.T)
            out = m + ex + em
            out = self.out_conv(out)
        else:
            out = m
       
        return out