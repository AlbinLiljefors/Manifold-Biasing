import torch
import torch.nn as nn
import torch.nn.functional as F

from lifting import FrozenLiftRaw
from iteration_block import FourPathIterBlock
from reconstruction import ReconstructionHead, compute_recon_loss
from analytic_filters import analytic_primary_circle, directional_derivative_filters

# Tuned SepLR config (from CV sweep)
MANIFOLD_RAW_SEPLR = {
    'joint_conv_lr': 5e-4,
    'iter_block_lr': 1e-7,
    'recon_lambda': 0.05,
    'weight_decay': 0.1,
}


def make_param_groups(model, config):
    return [
        {
            'filter': lambda name: 'iter_block' in name,
            'lr': config['iter_block_lr'],
            'weight_decay': config.get('weight_decay', 0.0),
        },
        {
            'filter': lambda name: 'joint_conv' in name,
            'lr': config['joint_conv_lr'],
        },
    ]


# All models share identical downstream; only first_layer differs.
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


class LoveFirstLayer(nn.Module):
    def __init__(self):
        super().__init__()
        filters = analytic_primary_circle(64, 3)
        self.register_buffer('filters', torch.from_numpy(filters).float().unsqueeze(1))

    def forward(self, x):
        return F.conv2d(x, self.filters)


class ManifoldFrozenFirstLayer(nn.Module):
    def __init__(self):
        super().__init__()
        filters = directional_derivative_filters(64, 3)
        self.register_buffer('filters', torch.from_numpy(filters).float().unsqueeze(1))

    def forward(self, x):
        return F.conv2d(x, self.filters)


class ManifoldRawFirstLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lift = FrozenLiftRaw()
        self.iter_block = FourPathIterBlock()
        self.joint_conv = nn.Conv2d(3, 64, 3)
        self.recon_head = ReconstructionHead()
        self._last_manifold = None

    def forward(self, x):
        gradients = self.lift(x)
        pixel, gradients = self.iter_block(x, gradients)
        self._last_manifold = gradients
        return self.joint_conv(torch.cat([pixel, gradients], dim=1))


class LoveNet(_BaseNet):
    def __init__(self):
        super().__init__()
        self.first_layer = LoveFirstLayer()


class ManifoldFrozenNet(_BaseNet):
    def __init__(self):
        super().__init__()
        self.first_layer = ManifoldFrozenFirstLayer()


class ManifoldRawNet(_BaseNet):
    def __init__(self):
        super().__init__()
        self.first_layer = ManifoldRawFirstLayer()

    def compute_recon_loss(self, image):
        return compute_recon_loss(
            self.first_layer._last_manifold, image, self.first_layer.recon_head
        )


class BaselineNet(_BaseNet):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Conv2d(1, 64, 3)


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    for buf in model.buffers():
        frozen += buf.numel()
    return trainable, frozen
