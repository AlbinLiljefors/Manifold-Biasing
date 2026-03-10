# Four-path iteration block (Manifold's spec):
#   pixel_new = ReLU(PP(pixel)) + MP(manifold)
#   manifold_new = PM(pixel) + MM(manifold)
# Identity init: PP,MM = identity; MP,PM = zero -> output ~ input at init.
import torch.nn as nn
import torch.nn.functional as F


class FourPathIterBlock(nn.Module):

    def __init__(self, manifold_ch):
        super().__init__()
        self.pp = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.mp = nn.Conv2d(manifold_ch, 1, 3, padding=1, bias=False)
        self.pm = nn.Conv2d(1, manifold_ch, 3, padding=1, bias=False)
        self.mm = nn.Conv2d(manifold_ch, manifold_ch, 3, padding=1, bias=False)
        self._init_weights(manifold_ch)

    def _init_weights(self, manifold_ch):
        nn.init.zeros_(self.pp.weight)
        self.pp.weight.data[0, 0, 1, 1] = 1.0
        nn.init.zeros_(self.mp.weight)
        nn.init.zeros_(self.pm.weight)
        nn.init.zeros_(self.mm.weight)
        for c in range(manifold_ch):
            self.mm.weight.data[c, c, 1, 1] = 1.0

    def forward(self, pixel, manifold):
        pixel_new = F.relu(self.pp(pixel)) + self.mp(manifold)
        manifold_new = self.pm(pixel) + self.mm(manifold)
        return pixel_new, manifold_new
