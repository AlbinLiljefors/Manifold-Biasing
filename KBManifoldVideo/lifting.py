# Frozen gradient lifting layers for K^t and T(K^t).
import math
import torch
import torch.nn as nn


class FrozenLiftKtNormalized(nn.Module):
    """Frozen 3x3 K^t lift (6 channels: Ix, Iy, Ixx, Ixy, Iyy, const)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, 3, padding=1, bias=False)

        NORM_Q2 = 2.0 / math.sqrt(3.0)
        NORM_Q3 = 1.0 / math.sqrt(5.0)

        K_Ix  = torch.tensor([[[-1, -1, -1],
                                [ 0,  0,  0],
                                [ 1,  1,  1]]]) / 6.0 * NORM_Q2

        K_Iy  = torch.tensor([[[-1,  0,  1],
                                [-1,  0,  1],
                                [-1,  0,  1]]]) / 6.0 * NORM_Q2

        K_Ixx = torch.tensor([[[ 1,  1,  1],
                                [-2, -2, -2],
                                [ 1,  1,  1]]]) / 6.0 * NORM_Q3

        K_Ixy = torch.tensor([[[ 1,  0, -1],
                                [ 0,  0,  0],
                                [-1,  0,  1]]]) / 4.0 * NORM_Q3

        K_Iyy = torch.tensor([[[ 1, -2,  1],
                                [ 1, -2,  1],
                                [ 1, -2,  1]]]) / 6.0 * NORM_Q3

        K_const = torch.ones(1, 3, 3) / 9.0

        self.conv.weight.data = torch.stack(
            [K_Ix, K_Iy, K_Ixx, K_Ixy, K_Iyy, K_const], dim=0)
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Temporal extension
# ---------------------------------------------------------------------------

class FrozenLiftTKt(nn.Module):
    """T(K^t) lift: 12 channels from 2 consecutive frames (6 spatial + 6 temporal)."""

    def __init__(self):
        super().__init__()
        NORM_Q2 = 2.0 / math.sqrt(3.0)
        NORM_Q3 = 1.0 / math.sqrt(5.0)

        K_Ix  = torch.tensor([[[-1, -1, -1],
                                [ 0,  0,  0],
                                [ 1,  1,  1]]]) / 6.0 * NORM_Q2
        K_Iy  = torch.tensor([[[-1,  0,  1],
                                [-1,  0,  1],
                                [-1,  0,  1]]]) / 6.0 * NORM_Q2
        K_Ixx = torch.tensor([[[ 1,  1,  1],
                                [-2, -2, -2],
                                [ 1,  1,  1]]]) / 6.0 * NORM_Q3
        K_Ixy = torch.tensor([[[ 1,  0, -1],
                                [ 0,  0,  0],
                                [-1,  0,  1]]]) / 4.0 * NORM_Q3
        K_Iyy = torch.tensor([[[ 1, -2,  1],
                                [ 1, -2,  1],
                                [ 1, -2,  1]]]) / 6.0 * NORM_Q3
        K_const = torch.ones(1, 3, 3) / 9.0

        kernels = torch.stack([K_Ix, K_Iy, K_Ixx, K_Ixy, K_Iyy, K_const], dim=0)

        self.spatial_conv = nn.Conv2d(1, 6, 3, padding=1, bias=False)
        self.spatial_conv.weight.data = kernels.clone()
        self.spatial_conv.weight.requires_grad_(False)

        self.temporal_conv = nn.Conv2d(1, 6, 3, padding=1, bias=False)
        self.temporal_conv.weight.data = kernels.clone()
        self.temporal_conv.weight.requires_grad_(False)

    def forward(self, x):
        frame_t = x[:, 0:1]
        D = frame_t - x[:, 1:2]
        spatial = self.spatial_conv(frame_t)
        temporal = self.temporal_conv(D)
        return torch.cat([spatial, temporal], dim=1)


class FrozenLiftTKtRaw(nn.Module):
    """T(K^t) lift with raw (un-normalized) monomials. 12 channels."""

    def __init__(self):
        super().__init__()

        K_Ix  = torch.tensor([[[-1, -1, -1],
                                [ 0,  0,  0],
                                [ 1,  1,  1]]]) / 6.0
        K_Iy  = torch.tensor([[[-1,  0,  1],
                                [-1,  0,  1],
                                [-1,  0,  1]]]) / 6.0
        K_Ixx = torch.tensor([[[ 1,  1,  1],
                                [-2, -2, -2],
                                [ 1,  1,  1]]]) / 6.0
        K_Ixy = torch.tensor([[[ 1,  0, -1],
                                [ 0,  0,  0],
                                [-1,  0,  1]]]) / 4.0
        K_Iyy = torch.tensor([[[ 1, -2,  1],
                                [ 1, -2,  1],
                                [ 1, -2,  1]]]) / 6.0
        K_const = torch.ones(1, 3, 3) / 9.0

        kernels = torch.stack([K_Ix, K_Iy, K_Ixx, K_Ixy, K_Iyy, K_const], dim=0)

        self.spatial_conv = nn.Conv2d(1, 6, 3, padding=1, bias=False)
        self.spatial_conv.weight.data = kernels.clone()
        self.spatial_conv.weight.requires_grad_(False)

        self.temporal_conv = nn.Conv2d(1, 6, 3, padding=1, bias=False)
        self.temporal_conv.weight.data = kernels.clone()
        self.temporal_conv.weight.requires_grad_(False)

    def forward(self, x):
        frame_t = x[:, 0:1]
        D = frame_t - x[:, 1:2]
        spatial = self.spatial_conv(frame_t)
        temporal = self.temporal_conv(D)
        return torch.cat([spatial, temporal], dim=1)


class FrozenLiftTKtScaled(nn.Module):
    """T(K^t) lift with per-channel learnable scale+bias on temporal channels. 12 channels."""

    def __init__(self):
        super().__init__()
        self.base_lift = FrozenLiftTKt()
        self.temporal_scale = nn.Parameter(torch.ones(6))
        self.temporal_bias = nn.Parameter(torch.zeros(6))

    def forward(self, x):
        features = self.base_lift(x)
        spatial = features[:, :6]
        temporal = features[:, 6:]
        scale = self.temporal_scale.view(1, 6, 1, 1)
        bias = self.temporal_bias.view(1, 6, 1, 1)
        temporal = temporal * scale + bias
        return torch.cat([spatial, temporal], dim=1)


class FrozenLiftTKtMultiFrame(nn.Module):
    """N-frame T(K^t) lift via binomial finite differences. 6*N channels."""

    def __init__(self, N=5):
        super().__init__()
        self.N = N

        NORM_Q2 = 2.0 / math.sqrt(3.0)
        NORM_Q3 = 1.0 / math.sqrt(5.0)

        K_Ix  = torch.tensor([[[-1, -1, -1],
                                [ 0,  0,  0],
                                [ 1,  1,  1]]]) / 6.0 * NORM_Q2
        K_Iy  = torch.tensor([[[-1,  0,  1],
                                [-1,  0,  1],
                                [-1,  0,  1]]]) / 6.0 * NORM_Q2
        K_Ixx = torch.tensor([[[ 1,  1,  1],
                                [-2, -2, -2],
                                [ 1,  1,  1]]]) / 6.0 * NORM_Q3
        K_Ixy = torch.tensor([[[ 1,  0, -1],
                                [ 0,  0,  0],
                                [-1,  0,  1]]]) / 4.0 * NORM_Q3
        K_Iyy = torch.tensor([[[ 1, -2,  1],
                                [ 1, -2,  1],
                                [ 1, -2,  1]]]) / 6.0 * NORM_Q3
        K_const = torch.ones(1, 3, 3) / 9.0

        kernels = torch.stack([K_Ix, K_Iy, K_Ixx, K_Ixy, K_Iyy, K_const], dim=0)

        self.conv = nn.Conv2d(1, 6, 3, padding=1, bias=False)
        self.conv.weight.data = kernels.clone()
        self.conv.weight.requires_grad_(False)

        binom = torch.zeros(N, N)
        for k in range(N):
            for j in range(k + 1):
                sign = (-1.0) ** j
                coeff = 1.0
                for i in range(j):
                    coeff *= (k - i) / (i + 1)
                binom[k, j] = sign * coeff
        self.register_buffer('binom_coeffs', binom)

    def forward(self, x):
        B, N, H, W = x.shape
        channels = []
        for k in range(self.N):
            diff = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
            for j in range(k + 1):
                diff += self.binom_coeffs[k, j] * x[:, j:j+1]
            channels.append(self.conv(diff))
        return torch.cat(channels, dim=1)
