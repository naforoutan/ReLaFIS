import torch

def add_gaussian_noise(x, std=0.1):
    """Add Gaussian noise with given standard deviation."""
    noise = torch.randn_like(x) * std
    return x + noise