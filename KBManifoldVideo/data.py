# Data loading for video (frame-pair) experiments.
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

DIR_LEFT = 0
DIR_RIGHT = 1
DIR_UP = 2
DIR_DOWN = 3


class FramePairDataset(torch.utils.data.Dataset):
    """MNIST-like dataset -> frame pairs [2,28,28] via random translations."""

    def __init__(self, base_dataset, max_shift=3, seed=42):
        self.base = base_dataset
        self.max_shift = max_shift
        gen = torch.Generator().manual_seed(seed)
        N = len(base_dataset)
        self.shifts = torch.randint(-max_shift, max_shift + 1, (N, 2),
                                    generator=gen)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        dx, dy = self.shifts[idx].tolist()

        shifted = torch.zeros_like(image)
        src_y_start = max(0, -dx)
        src_y_end = min(28, 28 - dx)
        src_x_start = max(0, -dy)
        src_x_end = min(28, 28 - dy)
        dst_y_start = max(0, dx)
        dst_y_end = min(28, 28 + dx)
        dst_x_start = max(0, dy)
        dst_x_end = min(28, 28 + dy)
        shifted[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            image[:, src_y_start:src_y_end, src_x_start:src_x_end]

        pair = torch.cat([image, shifted], dim=0)
        return pair, label


class DirectionFramePairDataset(torch.utils.data.Dataset):
    """Frame pairs labelled by direction of motion (4 classes).

    Pre-shifts frame_{t-1} to decorrelate absolute position from the label —
    spatial-only models are forced to chance without this (they'd solve it via
    where-the-digit-is instead of how-it-moved).
    """

    def __init__(self, base_dataset, min_shift=2, max_shift=4, seed=42):
        self.base = base_dataset
        gen = torch.Generator().manual_seed(seed)
        N = len(base_dataset)
        self.directions = torch.randint(0, 4, (N,), generator=gen)
        self.magnitudes = torch.randint(min_shift, max_shift + 1, (N,), generator=gen)
        self.pre_shifts = torch.randint(-6, 7, (N, 2), generator=gen)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, _ = self.base[idx]
        direction = self.directions[idx].item()
        mag = self.magnitudes[idx].item()
        pre_dx, pre_dy = self.pre_shifts[idx].tolist()

        if direction == DIR_LEFT:
            dx, dy = 0, -mag
        elif direction == DIR_RIGHT:
            dx, dy = 0, mag
        elif direction == DIR_UP:
            dx, dy = -mag, 0
        else:
            dx, dy = mag, 0

        frame_prev = _shift_image(image, pre_dx, pre_dy)
        frame_t = _shift_image(image, pre_dx + dx, pre_dy + dy)

        pair = torch.cat([frame_t, frame_prev], dim=0)
        return pair, direction


def _shift_image(image, dx, dy):
    shifted = torch.zeros_like(image)
    H, W = 28, 28
    src_y_start = max(0, -dx)
    src_y_end = min(H, H - dx)
    src_x_start = max(0, -dy)
    src_x_end = min(W, W - dy)
    dst_y_start = max(0, dx)
    dst_y_end = min(H, H + dx)
    dst_x_start = max(0, dy)
    dst_x_end = min(W, W + dy)
    shifted[:, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        image[:, src_y_start:src_y_end, src_x_start:src_x_end]
    return shifted


def load_moving_mnist(batch_size=128, data_dir='./data', max_shift=3, seed=42):
    """MNIST frame pairs for digit classification (10 classes)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_pairs = FramePairDataset(train_ds, max_shift=max_shift, seed=seed)
    test_pairs = FramePairDataset(test_ds, max_shift=max_shift, seed=seed + 1)

    train_loader = torch.utils.data.DataLoader(
        train_pairs, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_pairs, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_direction_mnist(batch_size=128, data_dir='./data', min_shift=2,
                         max_shift=4, seed=42):
    """MNIST frame pairs for direction classification (4 classes)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_pairs = DirectionFramePairDataset(
        train_ds, min_shift=min_shift, max_shift=max_shift, seed=seed)
    test_pairs = DirectionFramePairDataset(
        test_ds, min_shift=min_shift, max_shift=max_shift, seed=seed + 1)

    train_loader = torch.utils.data.DataLoader(
        train_pairs, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_pairs, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# KTH / Weizmann / Jester / SSv2 / UCF-101 frame pairs (.npz)
# ---------------------------------------------------------------------------

class NpzFramePairDataset(torch.utils.data.Dataset):
    """Dataset from preprocessed .npz file of frame pairs [N,2,H,W]."""

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.frames = torch.from_numpy(data['frames'])
        self.labels = torch.from_numpy(data['labels'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.frames[idx], self.labels[idx].item()


def _load_npz_pairs(train_path, test_path, batch_size, n_train, seed, missing_hint):
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(missing_hint)

    train_ds = NpzFramePairDataset(train_path)
    test_ds = NpzFramePairDataset(test_path)

    if n_train is not None and n_train < len(train_ds):
        gen = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(train_ds), generator=gen)[:n_train].tolist()
        train_ds = torch.utils.data.Subset(train_ds, indices)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_kth_pairs(batch_size=64, data_dir='./data', n_train=None, seed=42):
    """KTH frame pairs. Run preprocess_kth.py first."""
    return _load_npz_pairs(
        os.path.join(data_dir, 'kth_train.npz'),
        os.path.join(data_dir, 'kth_test.npz'),
        batch_size, n_train, seed,
        f"KTH data not found at {data_dir}. Run preprocess_kth.py first.")


def load_jester_pairs(batch_size=64, data_dir='./data', n_train=None, seed=42):
    """Jester 27-class frame pairs. Run preprocess_jester.py first."""
    return _load_npz_pairs(
        os.path.join(data_dir, 'jester_train.npz'),
        os.path.join(data_dir, 'jester_test.npz'),
        batch_size, n_train, seed,
        f"Jester data not found at {data_dir}. Run preprocess_jester.py first.")


def load_jester_direction_pairs(batch_size=64, data_dir='./data', n_train=None, seed=42):
    """Jester 22-class directional subset. Run preprocess_jester.py first."""
    return _load_npz_pairs(
        os.path.join(data_dir, 'jester_direction_train.npz'),
        os.path.join(data_dir, 'jester_direction_test.npz'),
        batch_size, n_train, seed,
        f"Jester direction data not found at {data_dir}. Run preprocess_jester.py first.")


def load_ssv2_pairs(batch_size=64, data_dir='./data', n_train=None, seed=42):
    """SSv2 174-class frame pairs. Run preprocess_ssv2.py first."""
    return _load_npz_pairs(
        os.path.join(data_dir, 'ssv2_train.npz'),
        os.path.join(data_dir, 'ssv2_test.npz'),
        batch_size, n_train, seed,
        f"SSv2 data not found at {data_dir}. Run preprocess_ssv2.py first.")


def load_ssv2_direction_pairs(batch_size=64, data_dir='./data', n_train=None, seed=42):
    """SSv2 directional subset. Run preprocess_ssv2.py first."""
    return _load_npz_pairs(
        os.path.join(data_dir, 'ssv2_direction_train.npz'),
        os.path.join(data_dir, 'ssv2_direction_test.npz'),
        batch_size, n_train, seed,
        f"SSv2 direction data not found at {data_dir}. Run preprocess_ssv2.py first.")


def load_ucf101_pairs(batch_size=64, data_dir='./data', n_train=None, seed=42):
    """UCF-101 frame pairs. Run preprocess_ucf101.py first."""
    return _load_npz_pairs(
        os.path.join(data_dir, 'ucf101_train.npz'),
        os.path.join(data_dir, 'ucf101_test.npz'),
        batch_size, n_train, seed,
        f"UCF-101 data not found at {data_dir}. Run preprocess_ucf101.py first.")


# ---------------------------------------------------------------------------
# Multi-frame (5-frame) data
# ---------------------------------------------------------------------------

class NpzMultiFrameDataset(torch.utils.data.Dataset):
    """Dataset from .npz file of multi-frame sequences [N,F,H,W]."""

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.frames = torch.from_numpy(data['frames'])
        self.labels = torch.from_numpy(data['labels'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.frames[idx], self.labels[idx].item()


def load_kth_5frame_pairs(batch_size=64, data_dir='./data', n_train=None, seed=42):
    """KTH 5-frame sequences at 90x60. Run preprocess_kth_5frame.py first."""
    train_path = os.path.join(data_dir, 'kth_5frame_train.npz')
    test_path = os.path.join(data_dir, 'kth_5frame_test.npz')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"KTH 5-frame data not found at {data_dir}. "
            f"Run preprocess_kth_5frame.py first.")

    train_ds = NpzMultiFrameDataset(train_path)
    test_ds = NpzMultiFrameDataset(test_path)

    if n_train is not None and n_train < len(train_ds):
        gen = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(train_ds), generator=gen)[:n_train].tolist()
        train_ds = torch.utils.data.Subset(train_ds, indices)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_weizmann_pairs(batch_size=64, data_dir='./data'):
    """Weizmann 2-frame test loader for KTH->Weizmann transfer."""
    test_path = os.path.join(data_dir, 'weizmann_test.npz')
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Weizmann data not found at {data_dir}. Run preprocess_weizmann.py first.")

    test_ds = NpzFramePairDataset(test_path)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False)
    return test_loader


def load_weizmann_5frame_pairs(batch_size=64, data_dir='./data'):
    """Weizmann 5-frame test loader at 90x60 for transfer with ResNet 5-frame models."""
    test_path = os.path.join(data_dir, 'weizmann_5frame_test.npz')
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Weizmann 5-frame data not found at {data_dir}. "
            f"Run preprocess_weizmann_5frame.py first.")
    test_ds = NpzMultiFrameDataset(test_path)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False)
    return test_loader
