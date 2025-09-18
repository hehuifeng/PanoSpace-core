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

from typing import TYPE_CHECKING
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


__all__ = [
    "detect_cells",
    "deconv_celltype", "superres_celltype", "celltype_annotator",
    "genexp_predictor",
]

# 给类型检查器正常导入（不影响运行时性能）
if TYPE_CHECKING:
    from .tl.detect import detect_cells
    from .tl.annotate import deconv_celltype, superres_celltype, celltype_annotator
    from .tl.predict import genexp_predictor

# 运行时真正的懒加载
def __getattr__(name: str):
    if name == "detect_cells":
        from .tl.detect import detect_cells
        return detect_cells
    if name in {"deconv_celltype", "superres_celltype", "celltype_annotator"}:
        from .tl.annotate import deconv_celltype, superres_celltype, celltype_annotator
        return {"deconv_celltype": deconv_celltype,
                "superres_celltype": superres_celltype,
                "celltype_annotator": celltype_annotator}[name]
    if name == "genexp_predictor":
        from .tl.predict import genexp_predictor
        return genexp_predictor
    raise AttributeError(f"module {__name__!r} has no attribute {name}")

def __dir__():
    return sorted(list(globals().keys()) + __all__)

