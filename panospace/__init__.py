"""
PanoSpace
=========
High-resolution spatial transcriptomics analysis toolkit built for the scverse
ecosystem.

Public API
----------
io     : Data I/O and format adapters.
tl     : High-level analysis functions (detect_cells, annotate_celltype, …).
pl     : Visualization helpers for spatial maps and networks.

Example
-------
>>> import panospace as ps
>>> cells = ps.tl.detect_cells(sdata)
"""
from __future__ import annotations

import importlib
import logging
import sys
import warnings
from importlib import metadata

from . import io, tl
# -----------------------------------------------------------------------------
# Version
# -----------------------------------------------------------------------------
try:
    __version__: str = metadata.version("panospace")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

# -----------------------------------------------------------------------------
# Logging (simple stdout handler; users can customise on their side)
# -----------------------------------------------------------------------------
_logger = logging.getLogger("panospace")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    )
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)

# Optional: silence known noisy warnings from third‑party libs
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# -----------------------------------------------------------------------------
# Lazy sub‑module loader (PEP 562)
# -----------------------------------------------------------------------------
__all__: list[str] = ["io", "tl", "pl", "__version__"]


def __getattr__(name: str):  # noqa: D401 – simple accessor
    """Dynamically import *io*, *tl* and *pl* only when accessed."""
    if name in {"io", "tl", "pl"}:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module  # cache in module globals for next time
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # noqa: D401 – complement built‑in dir()
    return sorted(list(globals().keys()) + ["io", "tl", "pl"])
