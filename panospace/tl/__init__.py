"""panospace.tl
===============
High-level *analysis* functions - mirroring the `scanpy.tl` naming convention -
that operate on :class:`spatialdata.SpatialData` objects and return updated
objects or dedicated result classes.  End-users are expected to import PanoSpace
as::

    >>> import panospace as ps
    >>> cells = ps.tl.detect_cells(img)

keeping all heavy lifting hidden behind these wrappers.

The sub-modules are lazily imported so that importing ``panospace`` costs almost
nothing until a specific function is accessed.
"""
from __future__ import annotations

import logging

from typing import Mapping

logger = logging.getLogger("panospace.tl")


_BACKENDS: Mapping[str, str] = {
    "predictor": "panospace._core.prediction.predictor:predictor_core",
    "RCTD": "panospace._core.annotation.RCTD:annotate_cells_core",
    "cell2location": "panospace._core.annotation.cell2location:annotate_cells_core",
    "spatialDWLS": "panospace._core.annotation.spatialDWLS:annotate_cells_core",
    "endecon": "panospace._core.annotation.endecon:endecon_core",
    "superres_core": "panospace._core.annotation.superres:superres_core",
    "annotator_core": "panospace._core.annotation.annotator:annotator_core",
}

# Validate backend imports during initialization
for backend, path in _BACKENDS.items():
    try:
        module_path, func_name = path.split(":")
        mod = __import__(module_path, fromlist=[func_name])
        getattr(mod, func_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import backend '{backend}' from '{path}': {e}")
        raise ImportError(f"Backend '{backend}' could not be imported. Check installation and dependencies.")
    
def _import_backend(name: str):
    """Dynamically import a backend function by name.

    Parameters
    ----------
    name : str
        Backend name, must be one of the keys in `_BACKENDS`.

    Returns
    -------
    callable
        The backend function object.

    Raises
    ------
    ValueError
        If the backend name is not registered.
    """
    if name not in _BACKENDS:
        raise ValueError(f"Unknown annotation backend '{name}'. Available: {list(_BACKENDS)}")
    module_path, func_name = _BACKENDS[name].split(":")
    mod = __import__(module_path, fromlist=[func_name])
    return getattr(mod, func_name)
