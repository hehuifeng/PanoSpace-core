"""panospace._core.annotation
===========================
Back-end implementations for **cell-type annotation** at single-cell
resolution.

Each concrete annotator - e.g. an *EnDecon*-based probabilistic mapper, a
*CytoSPACE* warping method, or any future approach - **must** live in its own
module inside this package *and* register itself under the *annotation* category
via :pyfunc:`panospace._core.register`::

    from panospace._core import register
    from typing import Any

    def deconv_celltype_core(adata: "AnnData", ref, **kwargs) -> "AnnData":
        ...  # heavy lifting here
        return adata

    register("annotation", "endecon", deconv_celltype_core)

Doing so makes the backend automatically discoverable by the public wrapper
:pyfunc:`panospace.tl.deconv_celltype`.
"""
from __future__ import annotations

from collections import defaultdict
import importlib
import logging
from types import ModuleType
from typing import Callable, Any, Dict, TYPE_CHECKING

from panospace._core import get as _get_backend, register as _register_backend

# if TYPE_CHECKING:  # pragma: no cover - static-type helpers only
#     from spatialdata import SpatialData  # type: ignore

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Lazy auto-import: attempt to import common back-ends so that they can register
# themselves. Failures are silenced (debug only) so that the package works even
# when optional deps are missing.
# -----------------------------------------------------------------------------
_AUTO_IMPORTS: Dict[str, str] = {
    "cell2location": ".cell2location",
    "RCTD": ".RCTD",
    "SpatialDWLS": ".SpatialDWLS",
    "endecon": ".endecon",
}

for _name, _rel_mod in _AUTO_IMPORTS.items():
    try:
        importlib.import_module(__name__ + _rel_mod)
        logger.debug("Annotation backend '%s' imported", _name)
    except ImportError as err:
        logger.debug("Could not import annotation backend '%s': %s", _name, err)

# -----------------------------------------------------------------------------
# Helper accessors - these are what higher-level tl wrappers should use.
# -----------------------------------------------------------------------------
BackendFunc = Callable[..., Any]

def get_backend(name: str) -> BackendFunc:
    """Return the registered annotation back-end *callable* for *name*.

    Parameters
    ----------
    name
        Key used during :pyfunc:`panospace._core.register`.

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    return _get_backend("annotation", name)  # type: ignore[return-value]


def available_backends() -> list[str]:
    """List names of all currently available annotation back-ends."""
    from panospace._core import available as _available  # local import

    return _available("annotation")

__all__ = [
    "get_backend",
    "available_backends",
]