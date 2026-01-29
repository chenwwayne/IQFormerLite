import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class VectorizedFastKANLayer(nn.Module):
    """
    多特征并行版本的 FastKANLayer。
    输入 (B, n_features)，输出 (B, n_features, d_embedding)。
    """
    def __init__(self, n_features, d_embedding, grid_min=-2., grid_max=2., num_grids=8, base_activation=nn.SiLU()):
        super().__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.base_activation = base_activation

        # base weight: 每个特征一个 d_embedding
        self.base_weight = nn.Parameter(torch.randn(n_features, d_embedding))

        # spline weight: 每个特征一个 spline
        self.spline_weight = nn.Parameter(torch.randn(n_features, d_embedding, num_grids))

        # 坐标网格
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.register_buffer("grid", grid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_features)
        return: (B, n_features, d_embedding)
        """
        # base part
        base_out = self.base_activation(x).unsqueeze(-1) * self.base_weight.unsqueeze(0)  # (B,n,d)

        # spline bases: (B, n, num_grids)
        spline_bases = torch.exp(-((x.unsqueeze(-1) - self.grid) ** 2))  # 简单RBF基，可替换为更复杂的
        spline_out = torch.einsum("bng,ndg->bnd", spline_bases, self.spline_weight)  # (B,n,d)

        return base_out + spline_out


class FastKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FastKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AttentionWithFastKANTransform(nn.Module):
    
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = True,
    ):
        super(AttentionWithFastKANTransform, self).__init__()

        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating
        self.linear_q = FastKANLayer(q_dim, total_dim)
        self.linear_k = FastKANLayer(k_dim, total_dim)
        self.linear_v = FastKANLayer(v_dim, total_dim)
        self.linear_o = FastKANLayer(total_dim, q_dim)
        self.linear_g = None
        if self.gating:
            self.linear_g = FastKANLayer(q_dim, total_dim)
        # precompute the 1/sqrt(head_dim)
        self.norm = head_dim**-0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor = None,      # additive attention bias
    ) -> torch.Tensor:         

        wq = self.linear_q(q).view(*q.shape[:-1], 1, self.num_heads, -1) * self.norm     # *q1hc
        wk = self.linear_k(k).view(*k.shape[:-2], 1, k.shape[-2], self.num_heads, -1)    # *1khc
        att = (wq * wk).sum(-1).softmax(-2)     # *qkh
        del wq, wk
        if bias is not None:
            att = att + bias[..., None]

        wv = self.linear_v(v).view(*v.shape[:-2],1, v.shape[-2], self.num_heads, -1)     # *1khc
        o = (att[..., None] * wv).sum(-3)        # *qhc
        del att, wv

        o = o.view(*o.shape[:-2], -1)           # *q(hc)

        if self.linear_g is not None:
            # gating, use raw query input
            g = self.linear_g(q)
            o = torch.sigmoid(g) * o

        # merge heads
        o = self.linear_o(o)
        return o

class KANEmbeddings(nn.Module):
    """
    矢量化版本的 KANEmbeddings，不再使用 for 循环。
    """
    def __init__(self, n_features, d_embedding, grid_size=8, grid_range=(-2., 2.), base_activation=nn.SiLU()):
        super().__init__()
        self.layer = VectorizedFastKANLayer(
            n_features=n_features,
            d_embedding=d_embedding,
            grid_min=grid_range[0],
            grid_max=grid_range[1],
            num_grids=grid_size,
            base_activation=base_activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_features)
        return: (B, n_features, d_embedding)
        """
        return self.layer(x)