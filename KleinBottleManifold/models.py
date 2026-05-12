import torch
import torch.nn as nn
import torch.nn.functional as F

from lifting import FrozenLiftKleinNormalized, FrozenLiftKtNormalized, FrozenLiftKtExact
from iteration_block import FourPathIterBlock
from reconstruction import ReconstructionHead, compute_recon_loss
from analytic_filters import klein_bottle_scaled_filters_nonuniform

SEPLR_CONFIG = {
    'joint_conv_lr': 1e-3,
    'iter_block_lr': 5e-6,
    'recon_lambda': 0.05,
    'weight_decay': 0.01,
}


# ---------------------------------------------------------------------------
# Shared downstream (Conv -> FC -> Softmax) only first_layer differs.
# ---------------------------------------------------------------------------

class _BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 24 * 24, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.first_layer(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# First layers 
# ---------------------------------------------------------------------------

class KleinRawFirstLayer(nn.Module):
    """Full Klein bottle K: 5ch lift -> iter_block -> joint_conv."""
    def __init__(self):
        super().__init__()
        self.lift = FrozenLiftKleinNormalized()
        self.iter_block = FourPathIterBlock(5)
        self.joint_conv = nn.Conv2d(6, 64, 3)
        self.recon_head = ReconstructionHead(5)
        self._last_manifold = None

    def forward(self, x):
        derivatives = self.lift(x)
        pixel = x
        pixel, derivatives = self.iter_block(pixel, derivatives)
        self._last_manifold = derivatives
        return self.joint_conv(torch.cat([pixel, derivatives], dim=1))


class KtRawFirstLayer(nn.Module):
    """Translational Klein K^t: 6ch lift -> iter_block -> joint_conv."""
    def __init__(self):
        super().__init__()
        self.lift = FrozenLiftKtNormalized()
        self.iter_block = FourPathIterBlock(6)
        self.joint_conv = nn.Conv2d(7, 64, 3)
        self.recon_head = ReconstructionHead(6)
        self._last_manifold = None

    def forward(self, x):
        features = self.lift(x)
        pixel = x
        pixel, features = self.iter_block(pixel, features)
        self._last_manifold = features
        return self.joint_conv(torch.cat([pixel, features], dim=1))


class KtExactRawFirstLayer(nn.Module):
    """K^t + exact spatial offset r: 7ch lift -> iter_block -> joint_conv."""
    def __init__(self):
        super().__init__()
        self.lift = FrozenLiftKtExact()
        self.iter_block = FourPathIterBlock(7)
        self.joint_conv = nn.Conv2d(8, 64, 3)
        self.recon_head = ReconstructionHead(7)
        self._last_manifold = None

    def forward(self, x):
        features = self.lift(x)
        pixel = x
        pixel, features = self.iter_block(pixel, features)
        self._last_manifold = features
        return self.joint_conv(torch.cat([pixel, features], dim=1))


class KleinFrozenFirstLayer(nn.Module):
    """64 frozen sin^2-weighted Klein bottle scaled filters."""
    def __init__(self, num_th1=8, num_th2=8):
        super().__init__()
        filters = klein_bottle_scaled_filters_nonuniform(num_th1, num_th2, m=3)
        self.register_buffer('filters', torch.from_numpy(filters).float().unsqueeze(1))

    def forward(self, x):
        return F.conv2d(x, self.filters)


# ---------------------------------------------------------------------------
# Full models (first layer + shared downstream)
# ---------------------------------------------------------------------------

class ManifoldKt(_BaseNet):
    def __init__(self):
        super().__init__()
        self.first_layer = KtRawFirstLayer()

    def compute_recon_loss(self, image):
        return compute_recon_loss(
            self.first_layer._last_manifold, image, self.first_layer.recon_head)


class ManifoldKtExact(_BaseNet):
    def __init__(self):
        super().__init__()
        self.first_layer = KtExactRawFirstLayer()

    def compute_recon_loss(self, image):
        return compute_recon_loss(
            self.first_layer._last_manifold, image, self.first_layer.recon_head)


class KleinFrozenNonUnif(_BaseNet):
    def __init__(self):
        super().__init__()
        self.first_layer = KleinFrozenFirstLayer()


class ManifoldKlein(_BaseNet):
    def __init__(self):
        super().__init__()
        self.first_layer = KleinRawFirstLayer()

    def compute_recon_loss(self, image):
        return compute_recon_loss(
            self.first_layer._last_manifold, image, self.first_layer.recon_head)


class Baseline(_BaseNet):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Conv2d(1, 64, 3)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def make_param_groups(model, config=None):
    if config is None:
        config = SEPLR_CONFIG
    return [
        {'filter': lambda name: 'iter_block' in name,
         'lr': config['iter_block_lr'],
         'weight_decay': config.get('weight_decay', 0.0)},
        {'filter': lambda name: 'joint_conv' in name,
         'lr': config['joint_conv_lr']},
    ]


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    for buf in model.buffers():
        frozen += buf.numel()
    return trainable, frozen


def get_model_specs():
    return [
        ('ManifoldKt',         ManifoldKt,          SEPLR_CONFIG['recon_lambda'], make_param_groups),
        ('ManifoldKtExact',    ManifoldKtExact,     SEPLR_CONFIG['recon_lambda'], make_param_groups),
        ('KleinFrozenNonUnif', KleinFrozenNonUnif,  0.0, None),
        ('ManifoldKlein',      ManifoldKlein,       SEPLR_CONFIG['recon_lambda'], make_param_groups),
        ('Baseline',                 Baseline,                   0.0, None),
    ]
