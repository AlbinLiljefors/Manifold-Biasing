# Four-path iteration blocks for K^t (6ch) and T(K^t) (12ch):
#   pixel_new = ReLU(PP(pixel)) + MP(manifold)
#   manifold_new = PM(pixel) + MM(manifold)
# Identity init: PP, MM = identity; MP, PM = zero.
import torch
import torch.nn as nn
import torch.nn.functional as F


class TKtNFourPathIterBlock(nn.Module):
    """Parameterized T(K^t) iteration block for N manifold channels."""

    def __init__(self, n_manifold=30):
        super().__init__()
        self.pp = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.mp = nn.Conv2d(n_manifold, 1, 3, padding=1, bias=False)
        self.pm = nn.Conv2d(1, n_manifold, 3, padding=1, bias=False)
        self.mm = nn.Conv2d(n_manifold, n_manifold, 3, padding=1, bias=False)
        self._init_weights(n_manifold)

    def _init_weights(self, n_manifold):
        nn.init.zeros_(self.pp.weight)
        self.pp.weight.data[0, 0, 1, 1] = 1.0
        nn.init.zeros_(self.mp.weight)
        nn.init.zeros_(self.pm.weight)
        nn.init.zeros_(self.mm.weight)
        for c in range(n_manifold):
            self.mm.weight.data[c, c, 1, 1] = 1.0

    def forward(self, pixel, manifold):
        pixel_new = F.relu(self.pp(pixel)) + self.mp(manifold)
        manifold_new = self.pm(pixel) + self.mm(manifold)
        return pixel_new, manifold_new
