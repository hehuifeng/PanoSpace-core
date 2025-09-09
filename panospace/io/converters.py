"""panospace.io.converters
========================
Utility helpers to *convert* between different on-disk representations and
in-memory objects once the raw data have been parsed by :pymod:`panospace.io.adapters`.

This module intentionally stays **format-agnostic**: it assumes the caller
already holds a fully initialised :pyclass:`spatialdata.SpatialData` object and
merely wants to serialise / de-serialise or extract sub-components for
inter-operability with other scverse or machine-learning tools.

Functions
---------
``save_zarr`` / ``load_zarr``
    Round-trip the entire SpatialData hierarchy to a directory Zarr store.
``cells_to_anndata``
    Extract the *cell* table plus expression layers and return an AnnData view -
    useful when you want to run Scanpy workflows on PanoSpace-predicted single
    cells.
``spots_to_anndata``
    Similar helper for spot-level tables (Visium etc.).
``merge_high_low``
    Combine a PanoSpace single-cell reconstruction with its original low-res
    SpatialData in a Common Coordinate System (CCS) so that Squidpy / napari can
    overlay both.
"""
from __future__ import annotations

from pathlib import Path
import logging
from typing import Union, Optional

import spatialdata as sdata
from spatialdata import SpatialData
from anndata import AnnData

__all__ = [
    "save_zarr",
    "load_zarr",
    "cells_to_anndata",
    "spots_to_anndata",
    "merge_high_low",
]

logger = logging.getLogger("panospace.io")

# -----------------------------------------------------------------------------
# Zarr serialisation helpers
# -----------------------------------------------------------------------------

def save_zarr(sd: SpatialData, path: Union[str, Path], *, overwrite: bool = False) -> Path:
    """Save *sd* to a Zarr directory.

    This is essentially a thin wrapper around ``sd.write`` with sensible defaults
    (consolidated metadata, compressor settings) and logging.
    """
    path = Path(path).with_suffix("")  # strip .zarr if not provided
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; use `overwrite=True` to replace.")

    logger.info("Writing SpatialData to %s ...", path)
    sd.write(path, consolidated=True)  # spatialdata>=0.1 provides this API
    logger.info("✓ Done - size on disk: %.2f MB", _dir_size_mb(path))
    return path


def load_zarr(path: Union[str, Path]) -> SpatialData:
    """Load a Zarr directory previously written by :pyfunc:`save_zarr`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    logger.info("Loading SpatialData from %s ...", path)
    sd = sdata.read(path)
    logger.info("✓ Loaded SpatialData with %d tables, %d images", len(sd.tables), len(sd.images))
    return sd


# -----------------------------------------------------------------------------
# Inter-operability helpers
# -----------------------------------------------------------------------------

def cells_to_anndata(sd: SpatialData, *, layer: str = "expr_pred", copy: bool = True) -> AnnData:
    """Return single-cell *AnnData* extracted from ``sd.tables['cells']``.

    Parameters
    ----------
    sd
        A *SpatialData* object containing a *cells* table produced by PanoSpace.
    layer
        Which layer to use as ``X``. Default is the predicted expression matrix
        (``expr_pred``). Use ``layer=None`` to keep X empty.
    copy
        Whether to copy (default) or view into the source arrays to save memory.
    """
    if "cells" not in sd.tables:
        raise KeyError("SpatialData object has no `cells` table. Run `ps.tl.detect_cells` first.")
    at = sd.tables["cells"]

    adata = at.to_anndata(X_name=layer, copy=copy)
    # carry over spatial coordinates in µm if available
    if "spatial" in adata.obsm:
        adata.obsm.setdefault("spatial_um", adata.obsm.pop("spatial"))
    adata.uns["platform"] = sd.tables["spots"].obs.get("platform", "unknown") if "spots" in sd.tables else "unknown"
    return adata


def spots_to_anndata(sd: SpatialData, *, layer: str | None = None, copy: bool = True) -> AnnData:
    """Extract the **spot** table as AnnData (mainly for Scanpy compatibility)."""
    if "spots" not in sd.tables:
        raise KeyError("SpatialData object has no `spots` table (low-res).")
    return sd.tables["spots"].to_anndata(X_name=layer, copy=copy)


# -----------------------------------------------------------------------------
# High-low fusion helper
# -----------------------------------------------------------------------------

def merge_high_low(
    low_res: SpatialData,
    high_res: SpatialData,
    *,
    ccs_name: str = "panospace_ccs",
    overwrite: bool = True,
) -> SpatialData:
    """Register *high_res* single-cell data into the coordinate system of *low_res*.

    Both inputs **must** already have spatial transforms set (handled by
    :pymod:`spatialdata_io`).  The function clones *low_res*, inserts the
    high-res tables/images and returns a new SpatialData object with a shared CCS.
    """
    import copy

    sdl = copy.deepcopy(low_res)  # avoid mutating callerʼs object

    # 1. Copy tables/images
    for k, t in high_res.tables.items():
        if k in sdl.tables and not overwrite:
            raise ValueError(f"Table '{k}' already exists in low_res.")
        sdl.tables[k] = t
    for k, im in high_res.images.items():
        if k in sdl.images and not overwrite:
            raise ValueError(f"Image '{k}' already exists in low_res.")
        sdl.images[k] = im

    # 2. Merge transformations - assume both use pixel→µm affine
    for name, tf in high_res.transformations.items():
        if name in sdl.transformations and not overwrite:
            raise ValueError(f"Transform '{name}' already exists.")
        sdl.transformations[name] = tf

    # 3. Set a common coordinate system label
    sdl.set_default_coordinate_system(ccs_name)
    logger.info(
        "Merged SpatialData: %d tables, %d images under CCS '%s'",
        len(sdl.tables), len(sdl.images), ccs_name,
    )
    return sdl

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _dir_size_mb(path: Path) -> float:
    """Return total directory size in MB."""
    return sum(p.stat().st_size for p in path.rglob("*")) / 1e6
