"""Shared category and entropy helpers for linguistic richness / ARE."""

import torch

N_LINGUISTIC_CATEGORIES = 4
RICHNESS_CEILING = 1.3862943611198906


def categories_from_two_branch(alpha1, alpha2, beta):
    """Map two-branch GIFT gates to equality/inequality/greater/less categories."""
    gaussian_branch = beta >= 0.5
    return torch.where(
        gaussian_branch,
        torch.where(alpha1 >= 0.5, torch.zeros_like(alpha1), torch.ones_like(alpha1)),
        torch.where(alpha2 >= 0.5, torch.full_like(alpha2, 2), torch.full_like(alpha2, 3)),
    ).long()


def categories_from_one_branch(alpha):
    """Map an equality/inequality gate to the shared four-category support."""
    return torch.where(alpha >= 0.5, torch.zeros_like(alpha), torch.ones_like(alpha)).long()


def richness_from_categories(category, rules_count, n_terms_per_rule, per_rule=False):
    """Return mean per-rule categorical entropy in nats, optionally with each rule."""
    eps = 1e-10
    entropies = torch.empty(rules_count, dtype=torch.float64, device=category.device)
    for rule in range(rules_count):
        counts = torch.bincount(category[:, rule], minlength=N_LINGUISTIC_CATEGORIES).double()
        probabilities = counts / n_terms_per_rule
        entropies[rule] = -(probabilities * torch.log(probabilities + eps)).sum()
    mean_richness = entropies.mean().item()
    return (mean_richness, entropies) if per_rule else mean_richness
