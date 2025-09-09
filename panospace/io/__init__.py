"""panospace.io
================
Data-access layer for PanoSpace: reading, writing and converting between
vendor-specific raw formats, **SpatialData**, and **AnnData**.

Public API
~~~~~~~~~~
- :func:`read_visium`
- :func:`read_xenium`
- :func:`to_spatialdata`
- :func:`as_anndata`

Lazy import rules follow the pattern used in :pymod:`scanpy.read`.
"""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict

# from .adapters import (
#     read_visium,
#     read_xenium,
#     read_cosmx,
#     read_stereo,
#     to_anndata,
#     to_spatialdata,
#     from_adata_and_image as from_adata, 
# )



__all__ = [
    "read_visium",
    "read_xenium",
    "read_cosmx",
    "read_stereo",
    "to_anndata",
    "to_spatialdata",
    "from_adata",  
]

# _LAZY_SUBMODULES maps public function name → (module_path, attr_name)
_LAZY_FUNCTIONS: Dict[str, tuple[str, str]] = {
    "read_visium": (".adapters", "read_visium"),
    "read_xenium": (".adapters", "read_xenium"),
    "read_cosmx": (".adapters", "read_cosmx"),
    "to_spatialdata": (".converters", "to_spatialdata"),
    "as_anndata": (".converters", "as_anndata"),
}

if TYPE_CHECKING:  # static type checkers resolve real objects
    from .adapters import read_visium, read_xenium, read_cosmx  # noqa: F401
    from .converters import as_anndata, to_spatialdata  # noqa: F401


def __getattr__(name: str) -> Any:  # pragma: no cover
    """Lazy loader for heavy I/O dependencies (pandas, rasterio, etc.)."""
    try:
        module_path, attr = _LAZY_FUNCTIONS[name]
    except KeyError as exc:  # attr not in map
        raise AttributeError(f"module 'panospace.io' has no attribute '{name}'") from exc

    module: ModuleType = import_module(__name__ + module_path)
    obj = getattr(module, attr)
    globals()[name] = obj  # cache for future lookups
    return obj
