# Frozen 3x3 Legendre-normalized gradient kernels for Klein bottle lifting
import math
import torch
import torch.nn as nn


class FrozenLiftKleinNormalized(nn.Module):

    def _build_kernels(self):
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

        return [K_Ix, K_Iy, K_Ixx, K_Ixy, K_Iyy]

    def __init__(self):
        super().__init__()
        kernels = self._build_kernels()
        self.conv = nn.Conv2d(1, len(kernels), 3, padding=1, bias=False)
        self.conv.weight.data = torch.stack(kernels, dim=0)
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


class FrozenLiftKtNormalized(FrozenLiftKleinNormalized):

    def _build_kernels(self):
        kernels = super()._build_kernels()
        kernels.append(torch.ones(1, 3, 3) / 9.0)  # zeroth-order mean extractor
        return kernels


class FrozenLiftKtExact(nn.Module):

    _INV_Q2 = 1.0 / (2.0 / math.sqrt(3.0))
    _INV_Q3 = 1.0 / (1.0 / math.sqrt(5.0))

    def __init__(self):
        super().__init__()
        self.base_lift = FrozenLiftKtNormalized()

    def forward(self, x):
        base = self.base_lift(x)

        # De-normalize to raw LS coefficients
        a1 = base[:, 0:1] * self._INV_Q2
        a2 = base[:, 1:2] * self._INV_Q2
        a3 = base[:, 2:3] * self._INV_Q3
        a4 = base[:, 3:4] * self._INV_Q3
        a5 = base[:, 4:5] * self._INV_Q3
        a0 = base[:, 5:6] * 9.0

        grad_mag = torch.sqrt((a1 ** 2 + a2 ** 2).clamp(min=1e-12))
        cos_th = a1 / grad_mag
        sin_th = a2 / grad_mag

        D2 = a3 * cos_th ** 2 + 2.0 * a4 * cos_th * sin_th + a5 * sin_th ** 2

        # Quadratic: D2*r^2 - grad_mag*r + (a0+D2) = 0
        disc = grad_mag ** 2 - 4.0 * D2 * (a0 + D2)
        sqrt_disc = torch.sqrt(disc.clamp(min=0.0))
        denom = 2.0 * D2
        denom = torch.where(denom.abs() < 1e-10, torch.ones_like(denom), denom)

        r_plus = (grad_mag + sqrt_disc) / denom
        r_minus = (grad_mag - sqrt_disc) / denom
        r = torch.where(r_plus.abs() <= r_minus.abs(), r_plus, r_minus)
        r = torch.where(torch.isfinite(r), r, torch.zeros_like(r)).clamp(-2.0, 2.0)

        return torch.cat([base, r], dim=1)
