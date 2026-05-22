import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from timm.models.layers import trunc_normal_



def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):

    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega = 1. / (10000 ** (omega / (embed_dim / 2)))
    out = pos[:, None] * omega[None, :]  # (M, D/2)

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):

    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0].reshape(-1))
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1].reshape(-1))
    emb = torch.cat([emb_h, emb_w], dim=1)
    return emb

def get_2d_sincos_pos_embed(embed_dim, H, W, cls_token=False, device='gpu'):
 
    grid_h = torch.arange(H, dtype=torch.float32, device=device)
    grid_w = torch.arange(W, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')  # (H,W)
    grid = torch.stack(grid, dim=0)  # (2, H, W)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)  # (H*W, embed_dim)

    if cls_token:
        cls_emb = torch.zeros((1, embed_dim), device=device)
        pos_embed = torch.cat([cls_emb, pos_embed], dim=0)
    return pos_embed



class CIM_MLP(nn.Module):
    def __init__(self, feat_dim=256, num_basis=128, num_samples=128):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_basis = num_basis
        self.num_samples = num_samples

        def proj_block(in_dim, out_dim):
            return nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_dim),
    
            )

        def linear_block(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.SiLU()
            )

        self.k = proj_block(feat_dim, feat_dim)
        self.q = nn.Linear(feat_dim, feat_dim)
        self.v = nn.Linear(feat_dim, feat_dim)

        self.q_token = nn.Parameter(torch.randn(num_samples, feat_dim))

        self.x_proj = linear_block(feat_dim, num_basis)
        self.out_proj = linear_block(feat_dim, feat_dim)

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.q_token, std=0.02)
        # Initialize all Linear weights properly
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)


    def forward(self, x: torch.Tensor, z: torch.Tensor):
        """
        Args:
            x: (B, C, H, W)
            z: (1, num_basis, C)
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # pos_embed = get_2d_sincos_pos_embed(self.feat_dim, H, W, False, x.device)  # (H*W, C)
        # pos_embed = pos_embed.unsqueeze(0).expand(B, -1, -1)  # (B, N, C)

        # Flatten spatial dimensions
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        #x_flat = x_flat + pos_embed
        # Query: (B, num_samples, C)
        q = self.q(self.q_token).unsqueeze(0).expand(B, -1, -1)  # (B, num_samples, C)

        # Key: (B, N, C)
        k = self.k(x_flat)  # (B, N, C)

        # Value: (B, num_basis, C)
        v = self.v(z).expand(B, -1, -1)  # (B, num_basis, C)

        # Attention: (B, num_samples, N)
        attn_scores = q @ k.transpose(1, 2)
        attn_probs = F.softmax(attn_scores / (C ** 0.5), dim=-1)
        # attn_weights = attn_probs @ x_flat #(B, num_samples, C)
        # attn_weights = attn_weights.mean(dim=1, keepdim=True) #(B, 1, C)
        attn_weights = attn_probs.mean(dim=1).unsqueeze(-1)  # (B, N, 1)

        # Weighted features: (B, N, C)
        x_weighted = x_flat * attn_weights  # (B, N, C)

        # Project to basis
        x_proj = self.x_proj(x_weighted)  # (B, N, num_basis)

        # Attend with basis values: (B, N, C)
        out = x_proj @ v  # (B, N, C)

        # Reshape to (B, C, H, W)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        out = self.out_proj(out.permute(0, 2, 3, 1).reshape(B * H * W, C))
        out = out.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        return out



class CIM_MLP_CN(nn.Module):
    def __init__(self, feat_dim=256, num_basis=128, num_samples=128):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_basis = num_basis
        self.num_samples = num_samples

        def proj_block(in_dim, out_dim):
            return nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_dim),
    
            )

        def linear_block(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.SiLU()
            )
        self.alpha_predictor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.SiLU(),
            nn.Linear(feat_dim, num_basis),
        )
        self.k = proj_block(feat_dim, feat_dim)
        self.q = nn.Linear(feat_dim, feat_dim)
        self.v = nn.Linear(feat_dim, feat_dim)

        self.q_token = nn.Parameter(torch.randn(num_samples, feat_dim))

        self.x_proj = linear_block(feat_dim, num_basis)
        self.out_proj = linear_block(feat_dim, feat_dim)

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.q_token, std=0.02)
        # Initialize all Linear weights properly
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)


    def forward(self, x: torch.Tensor, z: torch.Tensor):
        """
        Args:
            x: (B, C, H, W)
            z: (1, num_basis, C)
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # pos_embed = get_2d_sincos_pos_embed(self.feat_dim, H, W, False, x.device)  # (H*W, C)
        # pos_embed = pos_embed.unsqueeze(0).expand(B, -1, -1)  # (B, N, C)

        # Flatten spatial dimensions
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        #x_flat = x_flat + pos_embed
        # Query: (B, num_samples, C)
        q = self.q(self.q_token).unsqueeze(0).expand(B, -1, -1)  # (B, num_samples, C)

        # Key: (B, N, C)
        k = self.k(x_flat)  # (B, N, C)

        # Value: (B, num_basis, C)
        v = self.v(z).expand(B, -1, -1)  # (B, num_basis, C)

        # Attention: (B, num_samples, N)
        attn_scores = q @ k.transpose(1, 2)
        attn_probs = F.softmax(attn_scores / (C ** 0.5), dim=-1)

        sampled_feat = attn_probs @ x_flat #(B, num_samples, C)
        alpha = self.alpha_predictor(sampled_feat) #(B, num_samples, num_basis)
        alpha = F.softmax(alpha, dim=-1)  #(B, num_samples, num_basis)

        samples = alpha @ v  #(B, num_samples, C)
        samples = samples.mean(dim=1) #(B, C)
        # Weighted features: (B, N, C)
        out = self.out_proj(samples).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        return out


class MultiScaleBasisBlock(nn.Module):
    def __init__(self, in_channels, conv_cfg=dict(type='Conv2d'), norm_cfg=dict(type='GN', num_groups=32), **kwargs):
        super(MultiScaleBasisBlock, self).__init__()
        self.input_dim = in_channels
        self.basis_dim = in_channels
        self.num_basis = in_channels

        # define basis of X
        x_basis = torch.randn(self.num_basis//2, self.basis_dim)  # (K, d)
        Q, _ = torch.linalg.qr(x_basis.T) 
        orthogonal_basis = Q.T 
        self.x_basis = nn.Parameter(orthogonal_basis) 

        # define basis of M
        m_basis = torch.randn(self.num_basis//2, self.basis_dim)  # (K, d)  #4
        Q, _ = torch.linalg.qr(m_basis.T) 
        m_orthogonal_basis = Q.T 
        self.m_basis = nn.Parameter(m_orthogonal_basis) 


        self.ei  = CIM_MLP_CN(feat_dim= self.input_dim, num_basis=self.num_basis//2, num_samples=128)
        self.emi = CIM_MLP_CN(feat_dim= self.input_dim, num_basis=self.num_basis//2, num_samples=128)
  
        self.x_proj = ConvModule(self.input_dim, self.input_dim, 3, stride=1, padding=1, conv_cfg=conv_cfg,norm_cfg=norm_cfg, act_cfg=dict(type='SiLU'))
        self.m_proj = ConvModule(self.input_dim, self.input_dim, 3, stride=1, padding=1, conv_cfg=conv_cfg,norm_cfg=norm_cfg, act_cfg=dict(type='SiLU'))

    
        self.gamma = nn.Parameter(torch.zeros(1)) # None
        self.fuse_proj = ConvModule(self.input_dim, self.input_dim, 1, stride=1, 
                                    conv_cfg=conv_cfg,norm_cfg=norm_cfg, act_cfg=dict(type='SiLU'))
        
        self.out_proj = ConvModule(self.input_dim, self.input_dim, 3, stride=1, padding=1,conv_cfg=conv_cfg,norm_cfg=norm_cfg, act_cfg=dict(type='SiLU'))
       
        self.init_weights()
    def init_weights(self):

        for module in [self.x_proj, self.m_proj, self.fuse_proj,self.out_proj]:
            conv = module.conv
            if isinstance(conv, nn.Conv2d):
                nn.init.kaiming_normal_(conv.weight, mode='fan_out')
                if conv.bias is not None:
                    nn.init.zeros_(conv.bias)

        


    def ssp(self, x, V):
        # Prepare basis: V ∈ ℝ^{d × K}
        B, C, H, W = x.shape

        x_vec = x.permute(0, 2, 3, 1).reshape(-1, self.basis_dim)  # (B*H*W, d)
       # V = self.basis.T  # (d, K)

        VtV = (V.T @ V).float()  # (K, K)
     
        VtV_pinv = torch.linalg.inv(VtV)   #

        P = V @ VtV_pinv @ V.T  # (d, d)

        x_proj = x_vec @ P  # (B*H*W, d)

        x_out = x_proj.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B, d, H, W)


        return x_out

    def forward(self, x):


        x_out = self.ssp(x, self.x_basis.T)

        m = x_out + self.x_proj(x)

        m_out = self.ssp(m, self.m_basis.T)

        m_out = m_out + self.m_proj(m)

        ei = self.ei(x, self.x_basis)
        emi = self.emi(m, self.m_basis)
        
        x_out = self.fuse_proj(ei + emi) * self.gamma + m_out

        out = self.out_proj(x_out)
        
        return out
    