import torch
import torch.nn as nn


class JacobiKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0):
        super(JacobiKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        # 参数矩阵 (in, out, degree+1)
        self.jacobi_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )
        nn.init.normal_(
            self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1))
        )

        # 预先计算递推系数并存 buffer，避免 forward 里动态算
        k = torch.arange(2, degree + 1, dtype=torch.float32)
        theta_k = (2 * k + a + b) * (2 * k + a + b - 1) / (2 * k * (k + a + b))
        theta_k1 = (2 * k + a + b - 1) * (a * a - b * b) / (
            2 * k * (k + a + b) * (2 * k + a + b - 2)
        )
        theta_k2 = (k + a - 1) * (k + b - 1) * (2 * k + a + b) / (
            k * (k + a + b) * (2 * k + a + b - 2)
        )
        self.register_buffer("theta_k", theta_k)
        self.register_buffer("theta_k1", theta_k1)
        self.register_buffer("theta_k2", theta_k2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape: (B, inputdim)
        x = x.view(-1, self.inputdim)
        x = torch.tanh(x)  # normalize to [-1, 1]

        B = x.shape[0]
        # 存放所有阶次 (B, in, degree+1)
        jacobi = x.new_zeros(B, self.inputdim, self.degree + 1)
        jacobi[:, :, 0] = 1.0
        if self.degree >= 1:
            jacobi[:, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2

        # 递推（用向量化避免 Python for 循环逐 batch/feature）
        for i in range(2, self.degree + 1):
            jacobi[:, :, i] = (
                (self.theta_k[i - 2] * x + self.theta_k1[i - 2]) * jacobi[:, :, i - 1]
                - self.theta_k2[i - 2] * jacobi[:, :, i - 2]
            )

        # 投影 (B, outdim)
        y = torch.einsum("bid,iod->bo", jacobi, self.jacobi_coeffs)
        return y


# -------------------------------------------------
# 与 rtdl.FTTransformer 兼容的封装
# -------------------------------------------------
class KANEmbeddings(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        degree: int = 4,
        a: float = 1.0,
        b: float = 1.0,
        **kw,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding

        # 每个特征独立一层
        self.layers = nn.ModuleList(
            [
                JacobiKANLayer(
                    input_dim=1, output_dim=d_embedding, degree=degree, a=a, b=b
                )
                for _ in range(n_features)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(
                f"Expect 2-D input with {self.n_features} cols, got {x.shape}"
            )
        outs = [self.layers[i](x[:, i : i + 1]) for i in range(self.n_features)]
        return torch.stack(outs, dim=1)  # (B, n_features, d_embedding)
