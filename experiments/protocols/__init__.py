"""Paper-specific experiment protocol runners."""

from experiments.protocols.admtsk import run_admtsk_protocol
from experiments.protocols.anfis import run_anfis_protocol
from experiments.protocols.common import ProtocolRunResult
from experiments.protocols.generic import run_generic_protocol
from experiments.protocols.unfis import run_unfis_protocol
from experiments.protocols.vsrp_anya import run_vsrp_anya_protocol

__all__ = [
    "ProtocolRunResult",
    "run_admtsk_protocol",
    "run_anfis_protocol",
    "run_efnn_nulluni_protocol",
    "run_generic_protocol",
    "run_unfis_protocol",
    "run_vsrp_anya_protocol",
]


def __getattr__(name: str):
    if name == "run_efnn_nulluni_protocol":
        from experiments.protocols.efnn_nulluni import run_efnn_nulluni_protocol

        return run_efnn_nulluni_protocol
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
