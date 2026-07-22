"""
Shared linguistic-richness utility.

Why this exists
----------------
Each model class previously computed `linguistic_richness()` against its
*own* category count (LitAnfis: 2, GIFT: 4), and external
reporting code (evaluate.py) normalised by each model's own ceiling
log(N_categories) to get a 0-100% reading. That percentage is NOT
comparable across models: 100% for a 2-category model (cap log(2)) and
100% for a 4-category model (cap log(4)) do not mean the same amount of
absolute linguistic richness.

This module fixes that by computing entropy on a SHARED, FIXED 4-category
support for every model:

    0: equal
    1: not-equal
    2: greater-than
    3: less-than

Models that don't have a relational (greater/less) branch simply never
populate categories 2/3 — their per-rule distribution puts zero mass
there structurally. That's not a limitation of the metric; it IS the
point: it shows up as a real, absolute richness gap rather than being
hidden by each model getting graded on its own curve.

Because every model is scored on the same support, the raw nats value
H in [0, log(4)] is now directly comparable: a GIFT rule with
H=0.96 is unambiguously richer than a LitAnfis rule with H=0.62 — no
"% of own max" caveat needed anymore.
"""

import torch

N_LINGUISTIC_CATEGORIES = 4          # shared, fixed support for ALL models
RICHNESS_CEILING = float(torch.log(torch.tensor(4.0)))  # log(4) ≈ 1.3863


def categories_from_two_branch(alpha1, alpha2, beta):
    """
    Hard-assign each (feature_or_rank, rule) entry to one of the 4 shared
    categories, for models with BOTH a Gaussian (equal/not-equal) branch
    and a relational (greater/less) branch, mixed by beta.

    alpha1, alpha2, beta : tensors of identical shape (e.g. (in_features, rules))
        alpha1 : sigmoid(literal) — equal (>=0.5) vs not-equal (<0.5)
        alpha2 : sigmoid(temp)    — greater (>=0.5) vs less (<0.5)
        beta   : sigmoid(comb_weight) — Gaussian branch (>=0.5) vs relational branch

    Returns
    -------
    LongTensor, same shape, values in {0, 1, 2, 3}.
    """
    gaussian_branch = beta >= 0.5
    category = torch.where(
        gaussian_branch,
        torch.where(alpha1 >= 0.5,
                    torch.zeros_like(alpha1),
                    torch.ones_like(alpha1)),
        torch.where(alpha2 >= 0.5,
                    torch.full_like(alpha2, 2.0),
                    torch.full_like(alpha2, 3.0)),
    ).long()
    return category


def categories_from_one_branch(alpha1):
    """
    Hard-assign each (feature_or_rank, rule) entry to one of the 4 shared
    categories, for models with ONLY a Gaussian (equal/not-equal) branch
    and no relational (greater/less) branch.

    Categories 2 (greater-than) and 3 (less-than) are structurally
    unreachable for these models — every entry lands in {0, 1} only.
    This is intentional: it is what makes the resulting entropy directly
    comparable (and, by construction, capped lower) against models that
    do have a relational branch.

    alpha1 : tensor, e.g. (in_features, rules)
        sigmoid(literal) — equal (>=0.5) vs not-equal (<0.5)

    Returns
    -------
    LongTensor, same shape, values in {0, 1}.
    """
    category = torch.where(
        alpha1 >= 0.5,
        torch.zeros_like(alpha1),
        torch.ones_like(alpha1),
    ).long()
    return category


def richness_from_categories(category, rules_count, n_terms_per_rule, per_rule=False):
    """
    Given a (n_terms_per_rule, rules_count) category tensor with values in
    {0, 1, 2, 3} (categories 2/3 may simply never occur for some models),
    compute the Shannon entropy of the empirical category distribution
    per rule, on the shared 4-category support, then average over rules.

    Returns
    -------
    float
        Mean entropy across rules (nats), in [0, log(4)].
    per_rule : bool
        If True, also return the per-rule entropy tensor of shape
        (rules_count,), as (mean, per_rule_H).
    """
    eps = 1e-10
    entropies = torch.empty(rules_count, dtype=torch.float64, device=category.device)

    for j in range(rules_count):
        counts = torch.bincount(category[:, j], minlength=N_LINGUISTIC_CATEGORIES).double()
        p = counts / n_terms_per_rule
        H = -(p * torch.log(p + eps)).sum()
        entropies[j] = H

    mean_richness = entropies.mean().item()

    if per_rule:
        return mean_richness, entropies
    return mean_richness