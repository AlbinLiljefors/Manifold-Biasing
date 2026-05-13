# Video CNN models — temporal extension of K^t for frame-pair classification.
import torch
import torch.nn as nn
import torch.nn.functional as F

from lifting import (FrozenLiftTKt, FrozenLiftKtNormalized,
                     FrozenLiftTKtRaw, FrozenLiftTKtScaled,
                     FrozenLiftTKtMultiFrame)
from iteration_block import TKtNFourPathIterBlock
from reconstruction import ReconstructionHead, compute_recon_loss


SEPLR_CONFIG = {
    'joint_conv_lr': 5e-4,
    'iter_block_lr': 1e-7,
    'recon_lambda': 0.05,
    'weight_decay': 0.1,
}


# ---------------------------------------------------------------------------
# MNIST shared downstream (28x28). Only first_layer differs across subclasses.
# ---------------------------------------------------------------------------

class _BaseNet(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 24 * 24, 512)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):
        x = F.relu(self.first_layer(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# First layers
# ---------------------------------------------------------------------------

class TKtFirstLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lift = FrozenLiftTKt()
        self.iter_block = TKtNFourPathIterBlock(12)
        self.joint_conv = nn.Conv2d(13, 64, 3)
        self.recon_head = ReconstructionHead(12)
        self._last_manifold = None

    def forward(self, x):
        features = self.lift(x)
        pixel, features = self.iter_block(x[:, 0:1], features)
        self._last_manifold = features
        return self.joint_conv(torch.cat([pixel, features], dim=1))


class TKtRawFirstLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lift = FrozenLiftTKtRaw()
        self.iter_block = TKtNFourPathIterBlock(12)
        self.joint_conv = nn.Conv2d(13, 64, 3)
        self.recon_head = ReconstructionHead(12)
        self._last_manifold = None

    def forward(self, x):
        features = self.lift(x)
        pixel, features = self.iter_block(x[:, 0:1], features)
        self._last_manifold = features
        return self.joint_conv(torch.cat([pixel, features], dim=1))


class TKtScaledFirstLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lift = FrozenLiftTKtScaled()
        self.iter_block = TKtNFourPathIterBlock(12)
        self.joint_conv = nn.Conv2d(13, 64, 3)
        self.recon_head = ReconstructionHead(12)
        self._last_manifold = None

    def forward(self, x):
        features = self.lift(x)
        pixel, features = self.iter_block(x[:, 0:1], features)
        self._last_manifold = features
        return self.joint_conv(torch.cat([pixel, features], dim=1))


class KtFirstLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lift = FrozenLiftKtNormalized()
        self.iter_block = TKtNFourPathIterBlock(6)
        self.joint_conv = nn.Conv2d(7, 64, 3)
        self.recon_head = ReconstructionHead(6)
        self._last_manifold = None

    def forward(self, x):
        features = self.lift(x[:, 0:1])
        pixel, features = self.iter_block(x[:, 0:1], features)
        self._last_manifold = features
        return self.joint_conv(torch.cat([pixel, features], dim=1))


# ---------------------------------------------------------------------------
# MNIST models
# ---------------------------------------------------------------------------

class ManifoldTKt(_BaseNet):
    def __init__(self, n_classes=10):
        super().__init__(n_classes=n_classes)
        self.first_layer = TKtFirstLayer()

    def compute_recon_loss(self, image):
        return compute_recon_loss(
            self.first_layer._last_manifold, image[:, 0:1], self.first_layer.recon_head)


class ManifoldTKtRaw(_BaseNet):
    def __init__(self, n_classes=10):
        super().__init__(n_classes=n_classes)
        self.first_layer = TKtRawFirstLayer()

    def compute_recon_loss(self, image):
        return compute_recon_loss(
            self.first_layer._last_manifold, image[:, 0:1], self.first_layer.recon_head)


class ManifoldTKtScaled(_BaseNet):
    def __init__(self, n_classes=10):
        super().__init__(n_classes=n_classes)
        self.first_layer = TKtScaledFirstLayer()

    def compute_recon_loss(self, image):
        return compute_recon_loss(
            self.first_layer._last_manifold, image[:, 0:1], self.first_layer.recon_head)


class KtSpatialOnly(_BaseNet):
    def __init__(self, n_classes=10):
        super().__init__(n_classes=n_classes)
        self.first_layer = KtFirstLayer()

    def compute_recon_loss(self, image):
        return compute_recon_loss(
            self.first_layer._last_manifold, image[:, 0:1], self.first_layer.recon_head)


class Baseline(_BaseNet):
    def __init__(self, n_classes=10):
        super().__init__(n_classes=n_classes)
        self.first_layer = nn.Conv2d(2, 64, 3)


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
        ('ManifoldTKt',       ManifoldTKt,       0.05, make_param_groups),
        ('ManifoldTKtRaw',    ManifoldTKtRaw,    0.05, make_param_groups),
        ('ManifoldTKtScaled', ManifoldTKtScaled, 0.05, make_param_groups),
        ('KtSpatialOnly',     KtSpatialOnly,     0.05, make_param_groups),
        ('Baseline',          Baseline,          0.0,  None),
    ]


# ---------------------------------------------------------------------------
# Resolution-adaptive models for KTH (64x64)
# ---------------------------------------------------------------------------

class _VideoBaseNet(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.pool = nn.AdaptiveAvgPool2d(6)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, n_classes)

    def _downstream(self, x):
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def forward(self, x):
        return self._downstream(F.relu(self.first_layer(x)))


class ManifoldTKtVideoNet(_VideoBaseNet):
    def __init__(self, n_classes=6):
        super().__init__(n_classes=n_classes)
        self.first_layer = TKtFirstLayer()

    def compute_recon_loss(self, image):
        return compute_recon_loss(
            self.first_layer._last_manifold, image[:, 0:1], self.first_layer.recon_head)


class ManifoldKtVideoNet(_VideoBaseNet):
    def __init__(self, n_classes=6):
        super().__init__(n_classes=n_classes)
        self.first_layer = KtFirstLayer()

    def compute_recon_loss(self, image):
        return compute_recon_loss(
            self.first_layer._last_manifold, image[:, 0:1], self.first_layer.recon_head)


class BaselineVideoNet(_VideoBaseNet):
    def __init__(self, n_classes=6):
        super().__init__(n_classes=n_classes)
        self.first_layer = nn.Conv2d(2, 64, 3)


class BaselineSingleFrameVideoNet(_VideoBaseNet):
    def __init__(self, n_classes=6):
        super().__init__(n_classes=n_classes)
        self.first_layer = nn.Conv2d(1, 64, 3)

    def forward(self, x):
        return self._downstream(F.relu(self.first_layer(x[:, 0:1])))


def get_kth_model_specs():
    return [
        ('ManifoldTKt',         ManifoldTKtVideoNet,          0.05, make_param_groups),
        ('ManifoldKt',          ManifoldKtVideoNet,           0.05, make_param_groups),
        ('Baseline',            BaselineVideoNet,             0.0,  None),
        ('BaselineSingleFrame', BaselineSingleFrameVideoNet,  0.0,  None),
    ]


# ---------------------------------------------------------------------------
# Five-frame models for KTH (90x60)
# ---------------------------------------------------------------------------

class TKt5FrameFirstLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lift = FrozenLiftTKtMultiFrame(N=5)
        self.iter_block = TKtNFourPathIterBlock(n_manifold=30)
        self.joint_conv = nn.Conv2d(31, 64, 3)
        self.recon_head = ReconstructionHead(30)
        self._last_manifold = None

    def forward(self, x):
        features = self.lift(x)
        pixel, features = self.iter_block(x[:, 0:1], features)
        self._last_manifold = features
        return self.joint_conv(torch.cat([pixel, features], dim=1))


# ---------------------------------------------------------------------------
# 8-layer ResNet downstream for KTH 5-frame experiments
# ---------------------------------------------------------------------------

class _BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class _ResNet8VideoBaseNet(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.block1 = _BasicBlock(64, 64, stride=1)
        self.block2 = _BasicBlock(64, 128, stride=2)
        self.block3 = _BasicBlock(128, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, n_classes)

    def _downstream(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

    def forward(self, x):
        return self._downstream(F.relu(self.first_layer(x), inplace=True))


class TKt5FrameResNetVideoNet(_ResNet8VideoBaseNet):
    def __init__(self, n_classes=6):
        super().__init__(n_classes=n_classes)
        self.first_layer = TKt5FrameFirstLayer()

    def compute_recon_loss(self, image):
        return compute_recon_loss(
            self.first_layer._last_manifold, image[:, 0:1], self.first_layer.recon_head)


class Baseline5FrameResNetVideoNet(_ResNet8VideoBaseNet):
    def __init__(self, n_classes=6):
        super().__init__(n_classes=n_classes)
        self.first_layer = nn.Conv2d(5, 64, 3)


class BaselineSingleFrameResNetVideoNet(_ResNet8VideoBaseNet):
    def __init__(self, n_classes=6):
        super().__init__(n_classes=n_classes)
        self.first_layer = nn.Conv2d(1, 64, 3)

    def forward(self, x):
        return self._downstream(F.relu(self.first_layer(x[:, 0:1]), inplace=True))


def make_param_groups_resnet(model, config=None):
    if config is None:
        config = SEPLR_CONFIG
    bn_param_names = set()
    for mod_name, mod in model.named_modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for p_name, _ in mod.named_parameters(recurse=False):
                bn_param_names.add(f'{mod_name}.{p_name}' if mod_name else p_name)
    return [
        {'filter': lambda name: 'iter_block' in name,
         'lr': config['iter_block_lr'],
         'weight_decay': config.get('weight_decay', 0.0)},
        {'filter': lambda name: 'joint_conv' in name,
         'lr': config['joint_conv_lr']},
        {'filter': lambda name, _bn=bn_param_names: name in _bn,
         'lr': config['joint_conv_lr'],
         'weight_decay': 0.0},
    ]


def get_kth_5frame_resnet_model_specs():
    return [
        ('TKt5Frame',      TKt5FrameResNetVideoNet,          0.05, make_param_groups_resnet),
        ('Baseline5Frame', Baseline5FrameResNetVideoNet,     0.0,  make_param_groups_resnet),
        ('SingleFrame',    BaselineSingleFrameResNetVideoNet, 0.0, make_param_groups_resnet),
    ]
