# Reconstruction loss predict 3x3 patches from coordinate vectors, weighted by local contrast
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionHead(nn.Module):
    def __init__(self, n_coords):
        super().__init__()
        self.n_coords = n_coords
        self.linear = nn.Linear(n_coords, 9)

    def forward(self, coords):
        return self.linear(coords)


def compute_recon_loss(manifold, image, head):
    B, _, H, W = image.shape

    patches = F.unfold(image, kernel_size=3, padding=0)
    patches = patches.view(B, 9, H - 2, W - 2)

    _, _, pH, pW = patches.shape
    if manifold.shape[2] > pH:
        coords = manifold[:, :, 1:-1, 1:-1]
    else:
        coords = manifold

    coords_flat = coords.permute(0, 2, 3, 1).reshape(-1, head.n_coords)
    pred = head(coords_flat)
    pred = pred.view(B, pH, pW, 9).permute(0, 3, 1, 2)

    patch_mean = patches.mean(dim=1, keepdim=True)
    patches_centered = patches - patch_mean
    patch_norm = patches_centered.norm(dim=1, keepdim=True).clamp(min=1e-8)
    contrast = patches_centered.std(dim=1, keepdim=True)

    error = (pred - patches_centered / patch_norm) ** 2
    weighted_error = error * contrast
    return weighted_error.mean()
