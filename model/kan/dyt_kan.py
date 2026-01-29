import torch
import torch.nn as nn
import math

class RBFKANLayer(nn.Module):
    """
    RBFKANLayer (原 GPKANLayer): 
    基于径向基函数（Radial Basis Function）的 KAN 层。
    
    本质：
    输入 x 被映射到 num_grids 个基于 Tanh 近似的局部“钟形”基函数上。
    input -> (dist to grid) -> tanh_kernel -> weighted sum
    
    数学形式类似：
    phi(x) = gamma * tanh(alpha * (1 - 0.5 * ((x - grid)/l)^2)) + beta
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 64,
        length_scale: float = 1.0,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_grids = num_grids
        self.grid_min = grid_min
        self.grid_max = grid_max
        
        # 预计算常数: 0.5 / (l^2)
        self.length_scale = float(length_scale)
        self.register_buffer("inv_len_sq", torch.tensor(0.5 / (self.length_scale ** 2)))

        # Grid (基函数的中心点): (num_grids,)
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.register_buffer("grid", grid)

        # 可学习的核参数 (Shape: 1, input_dim, 1)
        # alpha 控制基函数的宽窄/陡峭程度
        self.alpha = nn.Parameter(torch.ones(1, input_dim, 1) * 0.5)
        self.gamma = nn.Parameter(torch.ones(1, input_dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, input_dim, 1))

        # 混合权重: 将 num_grids 个基函数的输出映射到 output_dim
        # (input_dim, num_grids, output_dim)
        self.rbf_weight = nn.Parameter(torch.empty(input_dim, num_grids, output_dim))
        
        self.init_scale = init_scale
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(
            self.rbf_weight, 
            mean=0.0, 
            std=self.init_scale, 
            a=-2 * self.init_scale, 
            b=2 * self.init_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim)
        return: (B, input_dim, output_dim)
        """
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected input shape (B, {self.input_dim}), got {x.shape}")

        # 1. 计算与 Grid 的距离平方
        # x: (B, D) -> (B, D, 1)
        # grid: (N,) -> (1, 1, N)
        # dist_sq: (B, D, N)
        x_expanded = x.unsqueeze(-1)
        grid_expanded = self.grid.view(1, 1, -1)
        dist_sq = (x_expanded - grid_expanded).pow(2)
        
        # 2. 计算基函数值 (DyT style Tanh Kernel)
        # 这是一个类高斯（Gaussian-like）的局部激活函数
        inner = 1.0 - dist_sq * self.inv_len_sq
        basis_features = torch.tanh(self.alpha * inner)
        basis_features = self.gamma * basis_features + self.beta  # (B, input_dim, num_grids)

        # 3. 聚合输出
        # (B, I, N) * (I, N, O) -> (B, I, O)
        return torch.einsum("bin,ino->bio", basis_features, self.rbf_weight)

    def extra_repr(self):
        return (f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
                f"num_grids={self.num_grids}, grid_range=({self.grid_min}, {self.grid_max})")


class KANEmbeddings(nn.Module):
    """
    最终嵌入层：Base Linear + RBF KAN
    """
    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        rbf_grid_min: float = -2.0,
        rbf_grid_max: float = 2.0,
        rbf_num_grids: int = 64,
        rbf_length_scale: float = 1.0,
        rbf_weight_init_scale: float = 0.1,
        scale_base: float = 1.0,
        base_activation=nn.SiLU,
        gate_bias_init: float = 0.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding
        self.scale_base = scale_base

        # Base 分支 (Linear-like)
        self.base_weight = nn.Parameter(torch.empty(n_features, d_embedding))
        self.base_activation = base_activation()

        # RBF 分支 (原 GP 分支)
        self.rbf_layer = RBFKANLayer(
            input_dim=n_features,
            output_dim=d_embedding,
            grid_min=rbf_grid_min,
            grid_max=rbf_grid_max,
            num_grids=rbf_num_grids,
            length_scale=rbf_length_scale,
            init_scale=rbf_weight_init_scale,
        )

        # Gate
        self.gate_logits = nn.Parameter(torch.full((n_features, 1), float(gate_bias_init)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        # rbf_layer 内部已初始化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_features)
        return: (B, n_features, d_embedding)
        """
        # Base Part
        base_out = self.base_activation(x).unsqueeze(-1) * self.base_weight.unsqueeze(0)

        # RBF Part
        rbf_out = self.rbf_layer(x)

        # Fusion
        gate = torch.sigmoid(self.gate_logits).view(1, self.n_features, 1)
        
        # gate * base + (1-gate) * rbf
        return torch.addcmul(rbf_out, base_out - rbf_out, gate)
