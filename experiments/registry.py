"""Map notebook / config model names to paper protocol IDs."""

from __future__ import annotations

from typing import Optional

PROTOCOL_EFNN_NULLUNI = "efnn_nulluni"
PROTOCOL_VSRP_ANYA = "vsrp_anya"
PROTOCOL_ADMTSK = "admtsk"
PROTOCOL_ANFIS = "anfis"
PROTOCOL_UNFIS = "unfis"
PROTOCOL_GENERIC = "generic"

# Canonical display names → protocol id.
_PROTOCOL_BY_ALIAS = {
    # EFNN-NullUni
    "efnn-nulluni": PROTOCOL_EFNN_NULLUNI,
    "efnn_nulluni": PROTOCOL_EFNN_NULLUNI,
    "efnnnulluni": PROTOCOL_EFNN_NULLUNI,
    "efnn": PROTOCOL_EFNN_NULLUNI,
    "efnn_null_uni": PROTOCOL_EFNN_NULLUNI,
    # VSRP-AnYa-EFS
    "vsrp-anya-efs": PROTOCOL_VSRP_ANYA,
    "vsrp_anya_efs": PROTOCOL_VSRP_ANYA,
    "vsrp-anya": PROTOCOL_VSRP_ANYA,
    "vsrp_anya": PROTOCOL_VSRP_ANYA,
    "vsrpanya": PROTOCOL_VSRP_ANYA,
    "vsrpanyaefs": PROTOCOL_VSRP_ANYA,
    # ADMTSK
    "admtsk": PROTOCOL_ADMTSK,
    "adm-tsk": PROTOCOL_ADMTSK,
    "adaptive_dombi_tsk": PROTOCOL_ADMTSK,
    # Classical ANFIS (Jang hybrid)
    "anfis": PROTOCOL_ANFIS,
    "classical_anfis": PROTOCOL_ANFIS,
    "sugeno_anfis": PROTOCOL_ANFIS,
    # UNFIS-c
    "unfis": PROTOCOL_UNFIS,
    "unfis_c": PROTOCOL_UNFIS,
    "unfisc": PROTOCOL_UNFIS,
}


def canonicalize_model_name(model_name: str) -> str:
    """Lowercase / strip separators for alias lookup."""
    key = str(model_name).strip().lower()
    for ch in (" ", "-", "."):
        key = key.replace(ch, "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key


def resolve_protocol_id(
    model_name: str,
    model_class: Optional[type] = None,
) -> str:
    """Return protocol id for a config key and/or model class.

    Prefer explicit name aliases; fall back to class-name heuristics and
    model flags so CLI / notebook typos still route correctly.
    """
    alias = canonicalize_model_name(model_name)
    # Compact form without underscores for keys like efnnnulluni.
    compact = alias.replace("_", "")
    if alias in _PROTOCOL_BY_ALIAS:
        return _PROTOCOL_BY_ALIAS[alias]
    if compact in _PROTOCOL_BY_ALIAS:
        return _PROTOCOL_BY_ALIAS[compact]

    if model_class is not None:
        cname = canonicalize_model_name(getattr(model_class, "__name__", ""))
        ccompact = cname.replace("_", "")
        if cname in _PROTOCOL_BY_ALIAS:
            return _PROTOCOL_BY_ALIAS[cname]
        if ccompact in _PROTOCOL_BY_ALIAS:
            return _PROTOCOL_BY_ALIAS[ccompact]
        # Class-name heuristics (flags are usually instance attrs only).
        if "efnn" in ccompact and "null" in ccompact:
            return PROTOCOL_EFNN_NULLUNI
        if "vsrp" in ccompact or ("anya" in ccompact and "efs" in ccompact):
            return PROTOCOL_VSRP_ANYA
        if "admtsk" in ccompact:
            return PROTOCOL_ADMTSK
        # Exact ANFIS class only — not LitAnfis / etc.
        if ccompact == "anfis":
            return PROTOCOL_ANFIS
        if ccompact == "unfis" or ccompact.startswith("unfis"):
            return PROTOCOL_UNFIS
        if getattr(model_class, "uses_admtsk_paper_training", None) is True:
            return PROTOCOL_ADMTSK
        if getattr(model_class, "uses_anfis_hybrid_training", None) is True:
            return PROTOCOL_ANFIS
        if getattr(model_class, "uses_efnn_nulluni_protocol", None) is True:
            return PROTOCOL_EFNN_NULLUNI
        if getattr(model_class, "uses_vsrp_anya_protocol", None) is True:
            return PROTOCOL_VSRP_ANYA
        if getattr(model_class, "uses_unfis_gqlm", None) is True:
            return PROTOCOL_UNFIS
        if getattr(model_class, "uses_unfis_protocol", None) is True:
            return PROTOCOL_UNFIS

    return PROTOCOL_GENERIC


def is_paper_protocol(protocol_id: str) -> bool:
    return protocol_id in {
        PROTOCOL_EFNN_NULLUNI,
        PROTOCOL_VSRP_ANYA,
        PROTOCOL_ADMTSK,
        PROTOCOL_ANFIS,
        PROTOCOL_UNFIS,
    }
