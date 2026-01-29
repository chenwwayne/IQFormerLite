import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # wavelet 参数
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))

        # 权重
        self.weight1 = nn.Parameter(torch.empty(out_features, in_features))
        self.wavelet_weights = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        # 批归一化
        self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x):
        # x: (B, in_features)
        B, in_f = x.shape
        # broadcasting: (B, out_f, in_f)
        x_scaled = (x.unsqueeze(1) - self.translation.unsqueeze(0)) / self.scale.unsqueeze(0)

        if self.wavelet_type == 'mexican_hat':
            wavelet = ((x_scaled ** 2) - 1) * torch.exp(-0.5 * x_scaled ** 2)
            wavelet *= (2 / (math.sqrt(3) * math.pi ** 0.25))

        elif self.wavelet_type == 'morlet':
            omega0 = 5.0
            wavelet = torch.exp(-0.5 * x_scaled ** 2) * torch.cos(omega0 * x_scaled)

        elif self.wavelet_type == 'dog':  # Derivative of Gaussian
            wavelet = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)

        elif self.wavelet_type == 'meyer':
            v = torch.abs(x_scaled)
            pi = math.pi

            def nu(t):  # smooth transition
                return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)

            meyer_aux = torch.where(
                v <= 0.5, 1.0,
                torch.where(v >= 1.0, 0.0, torch.cos(pi / 2 * nu(2 * v - 1)))
            )
            wavelet = torch.sin(pi * v) * meyer_aux

        elif self.wavelet_type == 'shannon':
            pi = math.pi
            wavelet = torch.sinc(x_scaled / pi)  # sinc
            # windowing
            window = torch.hamming_window(
                x_scaled.size(-1), periodic=False,
                dtype=x_scaled.dtype, device=x_scaled.device
            )
            wavelet = wavelet * window  # (broadcasting OK)

        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")

        # (B, out_f, in_f) * (1, out_f, in_f) -> sum over in_f
        wavelet_output = (wavelet * self.wavelet_weights.unsqueeze(0)).sum(dim=-1)
        return wavelet_output  # (B, out_features)

    def forward(self, x):
        wavelet_out = self.wavelet_transform(x)
        # 可以尝试加入 base_output
        # base_output = F.linear(self.base_activation(x), self.weight1)
        # combined = wavelet_out + base_output
        return self.bn(wavelet_out)


class KANEmbeddings(nn.Module):
    """连续特征 -> (B, n_features, d_embedding)"""
    def __init__(self, n_features, d_embedding, wavelet_type='mexican_hat', **kw):
        super().__init__()
        self.n_features = n_features
        self.d_embedding = d_embedding
        self.layers = nn.ModuleList([
            KANLinear(1, d_embedding, wavelet_type) for _ in range(n_features)
        ])

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"Expect 2-D input with {self.n_features} cols, got {x.shape}")
        outs = [layer(x[:, i:i+1]) for i, layer in enumerate(self.layers)]
        return torch.stack(outs, dim=1)  # (B, n_features, d_embedding)


class KAN(nn.Module):
    def __init__(self, layers_hidden, wavelet_type='mexican_hat'):
        super().__init__()
        self.layers = nn.ModuleList([
            KANLinear(in_f, out_f, wavelet_type)
            for in_f, out_f in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
