"""Verbalise learned GIFT antecedents as human-readable rules."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _unwrap_model(model):
    """Return the underlying ``nn.Module`` whether given a model or a wrapper."""
    if hasattr(model, "literal") and hasattr(model, "comb_weight"):
        return model
    if hasattr(model, "model"):
        return model.model
    raise ValueError(
        "Expected a GIFT-family nn.Module (with .literal/.temp/"
        ".comb_weight) or an sklearn wrapper exposing .model"
    )


def _default_names(count: int, prefix: str) -> List[str]:
    return [f"{prefix}{i}" for i in range(count)]


def clean_feature_name(name: str) -> str:
    """Strip pipeline prefixes and format a feature name for display.

    ``num_imputer__positive_auxillary_nodes`` -> ``Positive Auxillary Nodes``
    """
    base = name.split("__")[-1]
    return base.replace("_", " ").title()


def _fmt_geometry(center: float, sigma: float) -> Tuple[str, str]:
    """Plain and LaTeX parenthetical ``[m=…, σ=…]`` annotation."""
    plain = f"[m={center:.2f}, σ={sigma:.2f}]"
    latex = rf"[m={center:.2f},\ \sigma={sigma:.2f}]"
    return plain, latex


def _symbolic_label(rule_idx: int, feat_idx: int) -> Tuple[str, str]:
    """Return plain ``A(i,j)`` and LaTeX ``A_{j}^{i}`` (1-based rule *i*, feature *j*)."""
    i = rule_idx + 1
    j = feat_idx + 1
    plain = f"A({i},{j})"
    latex = rf"A_{{{j}}}^{{{i}}}"
    return plain, latex


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


def _dominant_class_name(model, rule_idx: int, class_names: Sequence[str]) -> str:
    bias = model.local_biases[rule_idx].detach().cpu()
    if getattr(model, "binary", False) or bias.numel() == 1:
        # Single logit: positive bias -> label 1, else label 0.
        class_idx = int((bias.reshape(-1)[0] > 0).item())
    else:
        class_idx = int(bias.argmax().item())
    if class_idx < len(class_names):
        return str(class_names[class_idx])
    return f"class_{class_idx}"


_REL_EQ = ("is", r"\textit{is}")
_REL_NEQ = ("is not", r"\textit{is not}")
_REL_GEQ = ("is at least", r"\textit{is at least}")
_REL_LEQ = ("is at most", r"\textit{is at most}")
_REL_RELAXED_PLAIN = "(relaxed — don't care)"
_REL_RELAXED_LATEX = r"(\textit{relaxed --- don't care})"


def _verbalize_feature(
    *,
    w1: float,
    w2: float,
    w3: float,
    center: float,
    sigma: float,
    rho: float,
    feature_name: str,
    rule_idx: int,
    feat_idx: int,
    threshold_w: float,
    relaxation_threshold: float = 0.8,
) -> Tuple[str, str]:
    """Return (plain_text, latex_fragment) for one feature in one rule."""
    feat_latex = _latex_escape(feature_name)
    label_plain, label_latex = _symbolic_label(rule_idx, feat_idx)
    geom_plain, geom_latex = _fmt_geometry(center, sigma)
    geom_suffix_plain = f" {geom_plain}"
    geom_suffix_latex = f"\ {geom_latex}"

    if rho > relaxation_threshold:
        plain = f"{feature_name} {_REL_RELAXED_PLAIN}{geom_suffix_plain}"
        latex = rf"\text{{{feat_latex}}}\ {_REL_RELAXED_LATEX}{geom_suffix_latex}"
        return plain, latex

    if w3 > 0.5:
        rel_plain, rel_latex = _REL_EQ if w1 > threshold_w else _REL_NEQ
    elif w2 > threshold_w:
        rel_plain, rel_latex = _REL_GEQ
    else:
        rel_plain, rel_latex = _REL_LEQ

    plain = f"{feature_name} {rel_plain} {label_plain}{geom_suffix_plain}"
    latex = rf"\text{{{feat_latex}}} {rel_latex} {label_latex}{geom_suffix_latex}"
    return plain, latex


def verbalize_rules(
    model,
    feature_names: Optional[Sequence[str]] = None,
    class_names: Optional[Sequence[str]] = None,
    threshold_w: float = 0.65,
    relaxation_threshold: float = 0.8,
) -> Tuple[str, str]:
    """Verbalise learned GIFT antecedents.

    For each rule ``i`` and feature ``j`` the function reads the post-sigmoid
    branch weights ``w1 = sigmoid(literal)``, ``w2 = sigmoid(temp)``,
    ``w3 = sigmoid(comb_weight)``, centre ``m_j^i``, width ``sigma_j^i``, and
    relaxation ``rho_j^i`` (the consequent-relaxation coefficient from
    ``relaxation_rate()``).

    Antecedent logic
    ----------------
    * ``rho_j^i > relaxation_threshold`` -> mark the term as relaxed / omitted.
    * ``w3 > 0.5`` (identity stream): ``is A_j^i`` if ``w1 > threshold_w`` else
      ``is not A_j^i``, where ``A_j^i`` is the symbolic label for the fuzzy set
      at rule ``i``, feature ``j`` (centre ``m_j^i`` in ``[m=…, σ=…]``).
    * ``w3 <= 0.5`` (order stream): ``is at least A_j^i`` if ``w2 > threshold_w``
      else ``is at most A_j^i``.

    Parameters
    ----------
    model:
        A trained :class:`model.GIFT.GIFT` (or sklearn wrapper).
    feature_names:
        Names for input features, indexed by feature position.
    class_names:
        Optional class labels. When given, a consequent column is added using
        the class with the largest learned rule bias.
    threshold_w:
        Dominance threshold for ``w1`` / ``w2`` (default 0.65).
    relaxation_threshold:
        ``rho`` above this value marks a feature as relaxed (default 0.8).

    Returns
    -------
    latex_table, plain_text
        A full LaTeX ``tabular`` environment and a plain-text equivalent with
        one rule per row.
    """
    model = _unwrap_model(model)
    if not (hasattr(model, "literal") and hasattr(model, "temp")
            and hasattr(model, "comb_weight")):
        raise ValueError(
            "Model does not expose literal/temp/comb_weight parameters."
        )

    n_features = model.in_features
    n_rules = model.rules_count

    if feature_names is None:
        feature_names = _default_names(n_features, "x")
    else:
        feature_names = list(feature_names)
        if len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must match "
                f"in_features ({n_features})."
            )

    feature_names = [clean_feature_name(n) for n in feature_names]

    eps = 1e-10
    one_over_e = 1.0 / np.e

    with torch.no_grad():
        w1 = torch.sigmoid(model.literal).cpu().numpy()          # (F, R)
        w2 = torch.sigmoid(model.temp).cpu().numpy()
        w3 = torch.sigmoid(model.comb_weight).cpu().numpy()
        centers = model.mean.cpu().numpy()
        sigmas = F.softplus(model.std).clamp(min=1e-3).cpu().numpy()

        H1 = -(w1 * np.log(w1 + eps)) / one_over_e
        H2 = -(w2 * np.log(w2 + eps)) / one_over_e
        rho = w3 * H1 + (1.0 - w3) * H2

    plain_rows: List[str] = []
    latex_rows: List[str] = []

    has_consequent = class_names is not None
    if has_consequent:
        class_names = list(class_names)
        is_binary = getattr(model, "binary", False) or model.out_features == 1
        expected = 2 if is_binary else model.out_features
        if len(class_names) != expected:
            raise ValueError(
                f"class_names length ({len(class_names)}) must match "
                f"{'2 for binary' if is_binary else 'out_features'} "
                f"({expected})."
            )

    for rule_idx in range(n_rules):
        plain_parts: List[str] = []
        latex_parts: List[str] = []

        for feat_idx in range(n_features):
            plain, latex = _verbalize_feature(
                w1=w1[feat_idx, rule_idx],
                w2=w2[feat_idx, rule_idx],
                w3=w3[feat_idx, rule_idx],
                center=centers[feat_idx, rule_idx],
                sigma=sigmas[feat_idx, rule_idx],
                rho=rho[feat_idx, rule_idx],
                feature_name=feature_names[feat_idx],
                rule_idx=rule_idx,
                feat_idx=feat_idx,
                threshold_w=threshold_w,
                relaxation_threshold=relaxation_threshold,
            )
            plain_parts.append(plain)
            latex_parts.append(latex)

        rule_label = f"Rule {rule_idx + 1}"
        rule_label_latex = rule_label.replace(" ", r"\ ")
        antecedent_plain = " AND ".join(plain_parts)
        antecedent_latex = r" \land ".join(latex_parts)

        if has_consequent:
            consequent = _dominant_class_name(model, rule_idx, class_names)
            plain_rows.append(f"{rule_label}: IF {antecedent_plain} THEN {consequent}")
            latex_rows.append(
                rf"${rule_label_latex}$ & "
                rf"${antecedent_latex}$ & \text{{{_latex_escape(consequent)}}} \\"
            )
        else:
            plain_rows.append(f"{rule_label}: IF {antecedent_plain}")
            latex_rows.append(
                rf"${rule_label_latex}$ & ${antecedent_latex}$ \\"
            )

    if has_consequent:
        header = r"\hline Rule & Antecedent & Consequent \\ \hline"
        col_spec = "cll"
    else:
        header = r"\hline Rule & Antecedent \\ \hline"
        col_spec = "cl"

    latex_table = (
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        f"{header}\n"
        + "\n".join(latex_rows)
        + "\n\\hline\n\\end{tabular}"
    )
    plain_text = "\n".join(plain_rows)

    return latex_table, plain_text
