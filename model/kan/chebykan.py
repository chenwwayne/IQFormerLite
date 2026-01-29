import torch
import torch.nn as nn
import math

class FastChebyKANLayer(nn.Module):
    """
    向量化版 Chebyshev KAN 层:
      - 支持所有 n_features 一次性计算
      - cheby_coeffs: (n_features, d_embedding, degree+1)
      - 计算 cos(arccos(tanh(x)) * k)
    """
    def __init__(self, n_features, d_embedding, degree: int = 4):
        super().__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding
        self.degree = degree

        # 参数: 每个 feature 独立 (n_features, d_embedding, degree+1)
        self.cheby_coeffs = nn.Parameter(
            torch.empty(n_features, d_embedding, degree + 1)
        )
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1.0 / (n_features * (degree + 1)))

        # 多项式阶数索引 buffer: (degree+1,)
        self.register_buffer("k", torch.arange(0, degree + 1, dtype=torch.get_default_dtype()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_features)
        return: (B, n_features, d_embedding)
        """
        # 归一化输入到 [-1, 1]
        x = torch.tanh(x)  # (B, n_features)

        # 计算 theta = arccos(x)
        theta = torch.acos(x).unsqueeze(-1)  # (B, n_features, 1)

        # cos(theta * k)
        basis = torch.cos(theta * self.k.view(1, 1, -1))  # (B, n_features, degree+1)

        # 加权: einsum over degree
        # coeffs: (n_features, d_embedding, degree+1)
        # basis:  (B, n_features, degree+1)
        # output: (B, n_features, d_embedding)
        y = torch.einsum("bnd,ndm->bnm", basis, self.cheby_coeffs.transpose(1, 2))
        return y


class KANEmbeddings(nn.Module):
    """
    连续特征逐特征 Chebyshev 嵌入
    输出: (B, n_features, d_embedding)
    """
    def __init__(self, n_features: int, d_embedding: int, degree: int = 4, **kw):
        super().__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding
        self.layer = FastChebyKANLayer(n_features, d_embedding, degree)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"Expect input (B, {self.n_features}), got {x.shape}")
        return self.layer(x)
