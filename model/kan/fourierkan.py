import torch
import torch.nn as nn
import math

class OptimizedFourierKANLayer(nn.Module):
    """
    优化版 Fourier KAN layer:
      - 频率 k 作为 buffer 只创建一次
      - fourier_coeffs 存储为 (2, n_features, gridsize, d_embedding)
      - 使用 einsum 计算加权和，减少临时张量
    输入: x (B, n_features)
    输出: y (B, n_features, d_embedding)
    """
    def __init__(self, n_features, d_embedding, gridsize, addbias=True, smooth_initialization=False, device=None, dtype=None):
        super().__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding
        self.gridsize = gridsize
        self.addbias = addbias

        # frequency buffer k: shape (gridsize,)
        k = torch.arange(1, gridsize + 1, dtype=torch.get_default_dtype())
        self.register_buffer("k", k)

        # grid normalization factor (用于初始化)：shape (gridsize,)
        if smooth_initialization:
            # higher frequencies smaller amplitude
            grid_norm_factor = (torch.arange(gridsize, dtype=torch.get_default_dtype()) + 1.0) ** 2.0
        else:
            grid_norm_factor = torch.full((gridsize,), math.sqrt(gridsize), dtype=torch.get_default_dtype())

        # store as buffer for init usage (not strictly necessary, but convenient)
        self.register_buffer("_init_norm", grid_norm_factor)

        # fourier_coeffs: shape (2, n_features, gridsize, d_embedding)
        # using normal init scaled by grid_norm_factor (broadcast)
        coeff_shape = (2, n_features, gridsize, d_embedding)
        # create a float tensor then register as Parameter
        coeff = torch.randn(coeff_shape, dtype=torch.get_default_dtype())
        # scale: broadcast over gridsize dimension
        coeff = coeff / (math.sqrt(n_features) * grid_norm_factor.view(1, 1, gridsize, 1))
        self.fourier_coeffs = nn.Parameter(coeff)

        if self.addbias:
            # bias shape (1, n_features, d_embedding) for easy broadcast
            self.bias = nn.Parameter(torch.zeros(1, n_features, d_embedding))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor):
        """
        x: (B, n_features)
        returns y: (B, n_features, d_embedding)
        """
        # x * k -> (B, n_features, gridsize)
        # ensure k on same device / dtype
        k = self.k.to(x.device).to(x.dtype)
        # broadcast multiplication: x.unsqueeze(-1) * k
        phase = x.unsqueeze(-1) * k.view(1, 1, -1)   # (B, n_features, gridsize)

        # trig
        cos_part = torch.cos(phase)  # (B, n_features, gridsize)
        sin_part = torch.sin(phase)  # (B, n_features, gridsize)

        # einsum with coeffs:
        # coeffs[0]: (n_features, gridsize, d_embedding)
        # we want: y_cos[b,n,d] = sum_k cos[b,n,k] * coeffs_cos[n,k,d]
        coeffs = self.fourier_coeffs  # (2, n_features, gridsize, d_embedding)
        y_cos = torch.einsum('bnk,nkd->bnd', cos_part, coeffs[0])
        y_sin = torch.einsum('bnk,nkd->bnd', sin_part, coeffs[1])

        y = y_cos + y_sin
        if self.addbias:
            y = y + self.bias.to(y.device)

        return y


class KANEmbeddings(nn.Module):
    """
    将你的 KANEmbeddings 中仅使用 base + fourier 的路径优化：
      - 用 OptimizedFourierKANLayer
      - 避免 forward 中不必要 spline 计算（如果不使用则删除/延迟）
    """
    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        grid_size=5,
        base_activation=nn.SiLU,
        fourier_gridsize=16,
        enable_spline=False,   # 如果你暂时不需要 spline，设为 False 可以节省大量计算/内存
    ):
        super().__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding
        self.grid_size = grid_size
        self.base_activation = base_activation()

        # Base transformation
        self.base_weight = nn.Parameter(torch.Tensor(n_features, d_embedding))

        # Fourier KAN Layer: 优化版
        self.fourier_layer = OptimizedFourierKANLayer(n_features, d_embedding, gridsize=fourier_gridsize)

        # 如果需要 spline（可选），在 enable_spline=True 时再初始化相关参数
        self.enable_spline = enable_spline
        if enable_spline:
            # 保留原有 spline 参数，但建议把 heavy init 放到 reset 或 lazy_init
            self.spline_weight = nn.Parameter(torch.Tensor(n_features, d_embedding, grid_size))
            # ... 你原来的 spline 初始化逻辑
        else:
            self.spline_weight = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        # x: (B, n_features)
        base_output = self.base_activation(x).unsqueeze(-1) * self.base_weight.unsqueeze(0)
        fourier_output = self.fourier_layer(x)
        out = base_output + fourier_output
        # 如果 later 需要 spline，再把 spline 部分加上
        return out
