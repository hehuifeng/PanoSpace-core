"""panospace._core.prediction
===========================
Back-end implementations for **single-cell gene-expression prediction / super-resolution**.

A concrete predictor - e.g. graph-based semi-supervised learning (*graph_ssl*),
a GNN super-pixel model (*gnn_superres*), or a diffusion-based approach - MUST
reside in its own module inside this package *and* register itself under the
*prediction* category via :pyfunc:`panospace._core.register`::

    from panospace._core import register

    def predict_expr_core(sdata: "SpatialData", **kwargs) -> "SpatialData":
        ...  # heavy lifting here
        return sdata

    register("prediction", "graph_ssl", predict_expr_core)

The light-weight wrappers in :pymod:`panospace.tl.predict` discover these
implementations via :pyfunc:`panospace._core.get` and expose a stable user API.

This sub-package auto-imports a small set of default back-ends if their
optional dependencies are available.  Missing dependencies will be reported at
*debug* level only and do **not** raise at import time, so that the rest of
PanoSpace can still be used in a minimal environment.
"""
from __future__ import annotations

import importlib
import logging
from typing import Callable, Dict, List, TYPE_CHECKING

from panospace._core import available as _available_global, get as _get_global

logger = logging.getLogger("panospace._core.prediction")

if TYPE_CHECKING:
    from spatialdata import SpatialData  # pragma: no cover

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
BackendFunc = Callable[..., "SpatialData"]

# -----------------------------------------------------------------------------
# Auto-import optional back-ends
# -----------------------------------------------------------------------------
_AUTO_IMPORTS: Dict[str, str] = {
    # name          module relative to this package
    "graph_ssl": ".graph_ssl",
    "gnn_superres": ".gnn_superres",
    "diffusion": ".diffusion",
}

for _name, _relmod in _AUTO_IMPORTS.items():
    try:
        importlib.import_module(__name__ + _relmod)
        logger.debug("Auto-imported prediction backend '%s' (%s)", _name, _relmod)
    except ModuleNotFoundError:  # pragma: no cover - optional dependency missing
        logger.debug("Prediction backend '%s' not available - optional dependency missing", _name)

# -----------------------------------------------------------------------------
# Convenience accessors (used by tl.predict)
# -----------------------------------------------------------------------------

def get_backend(name: str) -> BackendFunc:
    """Return a *registered* prediction backend by *name*.

    Raises
    ------
    KeyError
        If no backend with that name has been registered.
    """
    return _get_global("prediction", name)  # type: ignore[return-value]


def available_backends() -> List[str]:
    """Return the list of currently **available** prediction back-end names."""
    return _available_global("prediction")


__all__ = [
    "get_backend",
    "available_backends",
]
