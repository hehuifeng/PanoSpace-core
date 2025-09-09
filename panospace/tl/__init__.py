"""panospace.tl
===============
High-level *analysis* functions - mirroring the `scanpy.tl` naming convention -
that operate on :class:`spatialdata.SpatialData` objects and return updated
objects or dedicated result classes.  End-users are expected to import PanoSpace
as::

    >>> import panospace as ps
    >>> cells = ps.tl.detect_cells(sdata)

keeping all heavy lifting hidden behind these wrappers.

The sub-modules are lazily imported so that importing ``panospace`` costs almost
nothing until a specific function is accessed.
"""
from __future__ import annotations


import importlib
import logging
from types import ModuleType
from typing import Any, Dict

logger = logging.getLogger("panospace.tl")

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .detect import detect_cells
    from .annotate import deconv_celltype

# -----------------------------------------------------------------------------
# Mapping: public attribute name  ->  sub-module where it is implemented
# -----------------------------------------------------------------------------
_SUBMODULE_ATTRS: Dict[str, str] = {
    # detection
    "detect_cells": "detect",
    "CellDetectionResult": "detect",
    # annotation
    "deconv_celltype": "annotate",
    "AnnotationResult": "annotate",
    # gene expression prediction
    "predict_expr": "predict",
    # micro-environment analysis
    "microenv_analysis": "microenv",
}

__all__ = list(_SUBMODULE_ATTRS)


# -----------------------------------------------------------------------------
# Lazy attribute access
# -----------------------------------------------------------------------------

def __getattr__(name: str) -> Any:  # noqa: D401  (non imperative docstring)
    """Dynamically import sub-modules on first access.

    This keeps the initial import footprint minimal - most sub-modules depend on
    large libraries such as *torch* or *networkx*.  Once an attribute is
    resolved, it is cached in ``globals()`` for subsequent look-ups.
    """

    if name not in _SUBMODULE_ATTRS:
        raise AttributeError(f"module 'panospace.tl' has no attribute {name!r}")

    submod_name = _SUBMODULE_ATTRS[name]
    full_name = f"panospace.tl.{submod_name}"

    logger.debug("Lazy-loading %%s for attribute %%s", full_name, name)

    submod: ModuleType = importlib.import_module(full_name)
    attr = getattr(submod, name)
    globals()[name] = attr  # cache for future access
    return attr


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(__all__))
