# Reconstruction loss: predict 3x3 patches from (x, y), weighted by local contrast
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 9)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.linear(xy)


def compute_recon_loss(manifold: torch.Tensor, image: torch.Tensor,
                       head: ReconstructionHead) -> torch.Tensor:
    B, _, H, W = image.shape

    patches = F.unfold(image, kernel_size=3, padding=0).view(B, 9, H - 2, W - 2)
    _, _, pH, pW = patches.shape
    xy = manifold[:, :, 1:-1, 1:-1] if manifold.shape[2] > pH else manifold

    xy_flat = xy.permute(0, 2, 3, 1).reshape(-1, 2)
    pred = head(xy_flat).view(B, pH, pW, 9).permute(0, 3, 1, 2)

    patch_mean = patches.mean(dim=1, keepdim=True)
    patches_centered = patches - patch_mean
    patch_norm = patches_centered.norm(dim=1, keepdim=True).clamp(min=1e-8)
    contrast = patches_centered.std(dim=1, keepdim=True)

    error = (pred - patches_centered / patch_norm) ** 2
    return (error * contrast).mean()
