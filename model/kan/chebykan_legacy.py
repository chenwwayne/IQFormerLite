import torch
import torch.nn as nn


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y

# -------------------------------------------------
# 以下接口与 rtdl.FTTransformer 兼容
# -------------------------------------------------
class KANEmbeddings(nn.Module):
    """
    连续特征逐特征嵌入，输出 (B, n_features, d_embedding)。
    参数关键字保持与 fourierkan 一致，方便 rtdl 直接切换。
    """
    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        degree: int = 4,                 # Chebyshev 阶数，对应 grid_size 的作用
        **kw                               # 吸收其它无用关键字
    ):
        super().__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding

        # 每个特征独立一个 ChebyKANLayer：输入 1 -> 输出 d_embedding
        self.layers = nn.ModuleList([
            ChebyKANLayer(input_dim=1, output_dim=d_embedding, degree=degree)
            for _ in range(n_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"Expect 2-D input with {self.n_features} cols, got {x.shape}")
        # 逐特征切片并送入对应层
        outs = [self.layers[i](x[:, i:i+1]) for i in range(self.n_features)]
        return torch.stack(outs, dim=1)          # (B, n_features, d_embedding)