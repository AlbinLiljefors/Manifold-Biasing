import numpy as np
import torch


def generate_class_noise_params(num_classes=10, tau=0.2, omega_sq=0.04, seed=42):
    """Draw per-class (mu, sigma_sq) following Love et al. (JMLR 2023, Section 4.1.2)."""
    rng = np.random.RandomState(seed)
    mu = rng.normal(tau, np.sqrt(0.04), size=num_classes)
    sigma_sq = rng.chisquare(1, size=num_classes) * omega_sq
    return mu, sigma_sq


def add_class_noise(x, y, mu, sigma_sq):
    """Add class-correlated Gaussian noise to a batch."""
    noise = torch.zeros_like(x)
    for k in range(len(mu)):
        mask = (y == k)
        if mask.any():
            std_k = float(np.sqrt(sigma_sq[k]))
            noise[mask] = float(mu[k]) + std_k * torch.randn_like(x[mask])
    return x + noise
