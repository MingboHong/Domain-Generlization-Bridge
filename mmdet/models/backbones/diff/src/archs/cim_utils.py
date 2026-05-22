import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule



class ExpectationEstimatorToken(nn.Module):
    """
    Transformer-style Expectation Estimator for input (B, N, C)
    """
    def __init__(self, feat_dim=256, num_basis=128, num_samples=128, eps=1e-5, with_query=True):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_basis = num_basis
        self.num_samples = num_samples
        self.eps = eps
        self.with_query = with_query
        if self.with_query:
            # Linear layer to generate token weights
            self.q_proj = nn.Linear(feat_dim, num_samples)

        # Subspace basis for SSP
        basis = torch.randn(num_basis, feat_dim)
        Q, _ = torch.linalg.qr(basis.T)
        orthogonal_basis = Q.T
        self.register_parameter('basis', nn.Parameter(orthogonal_basis))

        # Output projection with LayerNorm + GELU
        self.out_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU()
        )

    def ssp(self, x, V):
        B, N, C = x.shape
        V = V.to(x.dtype).to(x.device)
        temp = x @ V.T                  # (B, N, K)
        K = V.shape[0]
        reg = self.eps * torch.eye(K, device=V.device, dtype=torch.float32)
        alpha_T = torch.linalg.solve((V @ V.T).float() + reg, temp.transpose(1, 2).float())  # (K, N, B)
        alpha = alpha_T.transpose(1, 2)                                     # (B, N, K)
        x_proj = alpha @ V                                                     # (B, N, C)
        return x_proj

    def forward(self, x):
        if self.with_query:
            samples_feats = self.q_proj(x)             # (B, N, S)
            weights = (samples_feats * F.softmax(samples_feats, dim=2)).sum(dim=2, keepdim=True)  # (B, N, 1)
            x_weighted = weights * x                   # (B, N, C)
        else:
            x_weighted = x

        x_proj = self.ssp(x_weighted, self.basis)  # (B, N, C)
        out = self.out_proj(x_proj)
        return out


class FrontDoorBasisBlock(nn.Module):
    """
    Transformer-style MultiScaleBasisBlock
    """
    def __init__(self, feat_dim, basis_reduction=2, basis_reduction_mode='div', with_ep=True, with_query=True, with_mssp=False):
        super().__init__()
        self.input_dim = feat_dim
        self.with_ep = with_ep
        self.with_mssp = with_mssp
        self.eps = 1e-5

        if basis_reduction_mode == 'sub':
            self.re_basis = feat_dim - basis_reduction
        elif basis_reduction_mode == 'div':
            self.re_basis = feat_dim // basis_reduction
        elif basis_reduction_mode == 'mul':
            self.re_basis = int(feat_dim * basis_reduction)
        # Linear projection instead of convolution
        self.m_proj = nn.Linear(feat_dim, feat_dim)
        if self.with_mssp:
            basis = torch.randn(self.re_basis, feat_dim)  # (K, C)
            Q, _ = torch.linalg.qr(basis.T)
            orthogonal_basis = Q.T  # (K, C)
            self.register_parameter('x_basis', nn.Parameter(orthogonal_basis))
        if with_ep:
            self.ex = ExpectationEstimatorToken(feat_dim=feat_dim, num_basis=self.re_basis, num_samples=feat_dim, with_query=with_query)
            self.em = ExpectationEstimatorToken(feat_dim=feat_dim, num_basis=self.re_basis, num_samples=feat_dim, with_query=with_query)

        self.out_proj = nn.Linear(feat_dim, feat_dim)

        self._init_weights()

    def ssp(self, x, V):
        B, N, C = x.shape
        V = V.to(x.dtype).to(x.device)
        temp = x @ V.T                  # (B, N, K)
        K = V.shape[0]
        reg = self.eps * torch.eye(K, device=V.device, dtype=torch.float32)
        alpha_T = torch.linalg.solve((V @ V.T).float() + reg, temp.transpose(1, 2).float())  # (K, N, B)
        alpha = alpha_T.transpose(1, 2)                                     # (B, N, K)
        x_proj = alpha @ V                                                     # (B, N, C)
        return x_proj

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, N, C)
        """
        m = self.m_proj(x)
        if self.with_ep:
            ex = self.ex(x)
            em = self.em(m)
            if self.with_mssp:
                m_out = self.ssp(x, self.x_basis)
            else:
                m_out = m
            out = m_out + ex + em
        else:
            out = m
        out = self.out_proj(out)
        return out