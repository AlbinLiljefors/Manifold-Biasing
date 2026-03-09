# Frozen 3x3 least-squares gradient kernels -> raw (dx, dy)
import torch
import torch.nn as nn


class FrozenLiftRaw(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1, bias=False)
        I = torch.tensor([[[-1, -1, -1],
                           [ 0,  0,  0],
                           [ 1,  1,  1]]]) / 6.0
        J = torch.tensor([[[-1,  0,  1],
                           [-1,  0,  1],
                           [-1,  0,  1]]]) / 6.0
        self.conv.weight.data = torch.stack([I, J], dim=0)
        self.conv.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
