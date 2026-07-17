"""Paper-specific and generic experiment protocols.

Use :func:`experiments.router.run_model_experiment` (or
:meth:`tests.evaluate.Evaluator` routing) so EFNN-NullUni, VSRP-AnYa-EFS,
ADMTSK, ANFIS, and UNFIS never fall through the generic Adam/CE path.
"""

from experiments.registry import (
    PROTOCOL_ADMTSK,
    PROTOCOL_ANFIS,
    PROTOCOL_EFNN_NULLUNI,
    PROTOCOL_GENERIC,
    PROTOCOL_UNFIS,
    PROTOCOL_VSRP_ANYA,
    canonicalize_model_name,
    resolve_protocol_id,
)
from experiments.router import run_model_experiment

__all__ = [
    "PROTOCOL_ADMTSK",
    "PROTOCOL_ANFIS",
    "PROTOCOL_EFNN_NULLUNI",
    "PROTOCOL_GENERIC",
    "PROTOCOL_UNFIS",
    "PROTOCOL_VSRP_ANYA",
    "canonicalize_model_name",
    "resolve_protocol_id",
    "run_model_experiment",
]
