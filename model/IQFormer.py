import os
import sys
import copy
import einops
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import math
from torch.autograd import Variable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "torch-conv-kan")))
from kan_convs.fast_kan_conv import FastKANConv1DLayer

KANConv1d = FastKANConv1DLayer

def stemIQ(in_chs, out_chs):
    """
    Stem Layer that is implemented by two layers of conv.
    Output: sequence of layers with final shape of [B, C, D]
    """
    return nn.Sequential(
        nn.Conv1d(in_chs, out_chs//2 , kernel_size=5, stride=1, padding=2,groups=in_chs),
        nn.BatchNorm1d(out_chs//2),
        )
    
def stemSTFT(f,in_chs, out_chs):
    """
    Stem Layer that is implemented by two layers of conv.
    Output: sequence of layers with final shape of [B, C, 1, D]
    """
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs//2 , kernel_size=(f,1), stride=1,groups=in_chs),
        nn.BatchNorm2d(out_chs//2),
        nn.ReLU())
    
class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, D]
    Output: tensor in shape [B, C, D/stride]
    """

    def __init__(self, patch_size=3, stride=1, padding=1,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm1d):
        super().__init__()
        patch_size = patch_size
        stride = stride
        padding = padding
        # User requested ONLY this to be KANConv1d
        # self.proj = KANConv1d(in_chans, embed_dim, kernel_size=patch_size,
        #                       stride=stride, padding=padding)
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class ConvEncoder_IQ(nn.Module):
    """
    Implementation of ConvEncoder with 3*3 and 1*1 convolutions.
    Input: tensor with shape [B, C, D]
    Output: tensor with shape [B, C, D]
    """

    def __init__(self, dim, hidden_dim=64, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(hidden_dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x
class FCN(nn.Module):
    """
    Implementation of FCN layer with 1*1 convolutions.
    Input: tensor with shape [B, C, D]
    Output: tensor with shape [B, C, D]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Conv1d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for IQFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1) #BxNxD
        key = torch.nn.functional.normalize(key, dim=-1) #BxNxD

        query_weight = query @ self.w_g # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor # BxNx1

        A = torch.nn.functional.normalize(A, dim=1) # BxNx1

        G = torch.sum(A * query, dim=1) # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        ) # BxNxD

        out = self.Proj(G * key) + query #BxNxD

        out = self.final(out) # BxNxD

        return out


class LocalRepresentation(nn.Module):
    """
    Local Representation module for IQFormer that is implemented by 3*3 depth-wise and point-wise convolutions.
    Input: tensor in shape [B, C, D]
    Output: tensor in shape [B, C, D]
    """

    def __init__(self, dim, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm1d(dim)
        self.pwconv1 = nn.Conv1d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x
    
class BandStem(nn.Module):
    def __init__(self, in_chs, out_chs):
        super().__init__()
        self.conv = nn.Conv1d(in_chs, out_chs, kernel_size=3, stride=1, padding=1, groups=1)
        self.bn = nn.BatchNorm1d(out_chs)
        self.act = nn.GELU()
        self.proj = nn.Conv1d(out_chs, out_chs, kernel_size=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.proj(x)
        return x

# class KANStem(nn.Module):
#     def __init__(self, in_chs, out_chs):
#         super().__init__()
#         self.conv = KANConv1d(in_chs, out_chs, kernel_size=7, stride=1, padding=3, groups=1)
#         self.bn = nn.BatchNorm1d(out_chs)
#         self.act = nn.GELU()
#         self.proj = nn.Conv1d(out_chs, out_chs, kernel_size=1)
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.act(x)
#         x = self.proj(x)
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class FilterbankKANStem(nn.Module):
    """
    STFT-like learnable filterbank with FastKANConv1DLayer:
    - long window (kernel_size=31 by default) to mimic STFT nperseg
    - produces band_k channels (frequency-band-like responses)
    - magnitude + log compression to mimic spectrogram energy
    - BN + 1x1 projection for stable fusion
    """
    def __init__(
        self,
        in_chs: int = 2,
        band_k: int = 32,
        kernel_size: int = 31,
        stride: int = 1,
        grid_size: int = 8,
        grid_range=(-2.0, 2.0),
        dropout: float = 0.0,
        base_activation=nn.SiLU,
        use_log_abs: bool = True,
        eps: float = 1e-6,
        post_proj: bool = True,
    ):
        super().__init__()
        ks = int(kernel_size)
        assert ks % 2 == 1, "kernel_size should be odd, so padding=ks//2 keeps length"
        pad = ks // 2

        self.use_log_abs = use_log_abs
        self.eps = eps

        # FastKANConv1DLayer signature:
        # (input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
        #  grid_size=8, base_activation=nn.SiLU, grid_range=[-2,2], dropout=0.0, norm_layer=nn.InstanceNorm1d, **norm_kwargs)
        self.fb = KANConv1d(
            input_dim=in_chs,
            output_dim=band_k,
            kernel_size=ks,
            groups=1,
            padding=pad,
            stride=stride,
            dilation=1,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=list(grid_range),
            dropout=dropout,
            norm_layer=nn.InstanceNorm1d,  # 默认就是这个，但写清楚
        )

        # 重要：把分布稳定下来，便于 Fusion 学到“用它”
        self.bn = nn.BatchNorm1d(band_k)

        # 可选：再加一个 1×1 投影 + BN（小成本，但提升稳定性）
        self.post_proj = post_proj
        if post_proj:
            self.proj = nn.Sequential(
                nn.GELU(),
                nn.Conv1d(band_k, band_k, kernel_size=1, bias=False),
                nn.BatchNorm1d(band_k),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 2, L)
        return: (B, band_k, L)
        """
        band = self.fb(x)  # (B, band_k, L)

        # 关键：做“能量化”非线性，逼近 |STFT| 的统计特性
        if self.use_log_abs:
            band = torch.log1p(torch.abs(band) + self.eps)
        # 你也可以试试更接近能量谱的版本：
        # band = torch.log1p(band.pow(2) + self.eps)

        band = self.bn(band)
        if self.post_proj:
            band = self.proj(band)
        return band


class Fusion(nn.Module):
    """
    IQFormer  Fusion Encoder Block.
    """

    def __init__(self, input_chanel, out_chanel, drop):

        super().__init__()

        self.Conv = nn.Sequential( nn.Conv1d(input_chanel,out_chanel,1),
                                  nn.BatchNorm1d(out_chanel),
                                  nn.GELU(),
                                  nn.Conv1d(out_chanel,out_chanel,1),
                                  
        )
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x, stft):
        fusion = self.Conv(torch.cat((x,stft),dim=1))
        return self.drop(fusion)

class IQFormer_Encoder(nn.Module):
    """
    IQFormer_Encoder Encoder Block for IQFormer. It consists of (1) Local representation module, (2) EfficientAdditiveAttention, and (3) FCN block.
    Input: tensor in shape [B, C, D]
    Output: tensor in shape [B, C, D]
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.local_representation = LocalRepresentation(dim=dim, kernel_size=3, drop_path=0.,
                                                                   use_layer_scale=True)
        self.attn = EfficientAdditiveAttnetion(in_dims=dim, token_dim=dim, num_heads=1)
        self.linear = FCN(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        x = self.local_representation(x)
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1 * self.attn(x.permute(0, 2, 1)).permute(0, 2, 1))
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x))

        else:
            x = x + self.drop_path(
                self.attn(x.permute(0, 2, 1)).permute(0, 2, 1))
            x = x + self.drop_path(self.linear(x))
        return x


def Stage(dim, index, layers, mlp_ratio=4.,
          act_layer=nn.GELU,
          drop_rate=.0, drop_path_rate=0.,
          use_layer_scale=True, layer_scale_init_value=1e-5, vit_num=1):
    """
    Implementation of each IQFormer stages. Here, IQFormerEncoder used as the last block in all stages, while ConvEncoder used in the rest of the blocks.
    Input: tensor in shape [B, C, D]
    Output: tensor in shape [B, C, D]
    """
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)

        if layers[index] - block_idx <= vit_num:
            blocks.append(IQFormer_Encoder(
                dim, mlp_ratio=mlp_ratio,
                act_layer=act_layer, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value))

        else:
            blocks.append(ConvEncoder_IQ(dim=dim, hidden_dim=int(mlp_ratio * dim), kernel_size=3))
    blocks = nn.Sequential(*blocks)
    return blocks


class IQFormer(nn.Module):

    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=4,
                 act_layer=nn.GELU,
                 num_classes=11,
                 down_patch_size=5, down_stride=3, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 vit_num=1,
                 aux_mode='none',
                 band_k=32,
                 kernel_size=31,
                 grid_size=2,
                 grid_range=(-2.0, 2.0)):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.aux_mode = aux_mode
        self.BN = nn.BatchNorm1d(2)
        
        if self.aux_mode == 'stft':
            self.BN_stft = nn.BatchNorm2d(1)
            self.patch_embedIQ = stemIQ(2, embed_dims[0]//4) # out: embed_dims[0]//8
            self.patch_embedSTFT = stemSTFT(32,1,embed_dims[0]//4) # out: embed_dims[0]//8
            # Fusion in: embed_dims[0]//4, out: embed_dims[0]
            self.fusion = Fusion(embed_dims[0]//4, embed_dims[0], drop_rate)
        elif self.aux_mode == 'conv':
            self.patch_embedIQ = stemIQ(2, embed_dims[0]//4) # out: embed_dims[0]//8
            # BandStem out: band_k
            self.bandstem = BandStem(2, band_k)
            # Fusion in: embed_dims[0]//8 + band_k, out: embed_dims[0]
            self.fusion = Fusion(embed_dims[0]//8 + band_k, embed_dims[0], drop_rate)
        elif self.aux_mode == 'kan':
            self.patch_embedIQ = stemIQ(2, embed_dims[0]//4) # out: embed_dims[0]//8
            # self.kanstem = KANStem(2, band_k)
            self.kanstem = FilterbankKANStem(
                in_chs=2,
                band_k=band_k,
                kernel_size=kernel_size,
                stride=1,            # hop=1
                grid_size=grid_size,
                grid_range=grid_range,
                dropout=0.0,         # 先关掉，确保可控；后续再试 0.1
                base_activation=nn.SiLU,
                use_log_abs=True,    # 强烈建议开
                post_proj=True,
            )
            # Fusion in: embed_dims[0]//8 + band_k, out: embed_dims[0]
            self.fusion = Fusion(embed_dims[0]//8 + band_k, embed_dims[0], drop_rate)
        else: # none
            # Direct IQ embedding to full dimension
            self.patch_embedIQ = stemIQ(2, embed_dims[0]*2) # out: embed_dims[0]

        network = []
        for i in range(len(layers)):
            stage = Stage(embed_dims[i], i, layers, mlp_ratio=mlp_ratios,
                          act_layer=act_layer,
                          drop_rate=drop_rate,
                          drop_path_rate=drop_path_rate,
                          use_layer_scale=use_layer_scale,
                          layer_scale_init_value=layer_scale_init_value,
                          vit_num=vit_num)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    Embedding(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )

        self.network = nn.ModuleList(network)
        # self.patch_LSTM = nn.LSTM(input_size=embed_dims[0]//2, hidden_size=embed_dims[0]//2,bidirectional=True, batch_first=True, num_layers=2, dropout=drop_rate)

        # Classifier head
        self.norm = nn.BatchNorm1d(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 \
            else nn.Identity()
        self.apply(self._init_weights)
        self.globalmaxpool = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
        )
        self.globalavgpool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d,nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x, stft=None):
        x = self.BN(x)
        
        if self.aux_mode == 'stft':
            if stft is None:
                raise ValueError("stft input is required for aux_mode='stft'")
            stft = self.BN_stft(stft)
            x_iq = self.patch_embedIQ(x)
            stft_feat = torch.squeeze(self.patch_embedSTFT(stft))
            x = self.fusion(x_iq, stft_feat)
        elif self.aux_mode == 'conv':
            x_iq = self.patch_embedIQ(x)
            # Generate band features from IQ
            # Note: BandStem takes (B, 2, L)
            band_feat = self.bandstem(x) 
            x = self.fusion(x_iq, band_feat)
        elif self.aux_mode == 'kan':
            x_iq = self.patch_embedIQ(x)
            kan_feat = self.kanstem(x)
            x = self.fusion(x_iq, kan_feat)
        else: # none
            x = self.patch_embedIQ(x)
            
        # x,_ = self.patch_LSTM(x.permute(0,2,1))
        x = self.forward_tokens(x)
        x = self.norm(x)
        cls_out = self.head(self.globalavgpool(x))
        return cls_out
