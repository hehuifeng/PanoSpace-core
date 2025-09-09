"""panospace.tl.detect
====================
User-facing *nuclei/cell detection* wrapper.  It delegates the heavy lifting to
backend implementations in :pymod:`panospace._core.detection` (CellViT,
StarDist, …) and writes the resulting cell table into the supplied
:class:`spatialdata.SpatialData` object.

Example
-------
>>> import panospace as ps
>>> sdata = ps.io.read_visium("/path/to/visium")
>>> sdata = ps.tl.detect_cells(sdata, model="cellvit", gpu=True)

After execution the table ``sdata.tables['cells']`` will contain at least the
columns ``['cell_id', 'x', 'y']`` as defined in :pymod:`panospace.io._schemas`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

# import spatialdata as sdata_mod  # type: ignore - only for type checking
from PIL import Image
Image.MAX_IMAGE_PIXELS = 10000000000
# from panospace.io.converters import _ensure_spatialdata  # type: ignore - private helper

logger = logging.getLogger("panospace.tl")

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def detect_cells(
    img: Image.Image,
    *,
    model: Literal["cellvit", "stardist"] = "cellvit",
    # **kwargs: Any,
):
    """Detect nuclei / cells on the high-resolution image.

    Parameters
    ----------
    sdata
        A :class:`spatialdata.SpatialData` object **or** a path to a directory/
        file that can be read by :pyfunc:`panospace.io.read_visium`,
        :pyfunc:`~panospace.io.read_xenium`, etc.
    model
        Detection backend to use - currently ``"cellvit"`` (default) or
        ``"stardist"``.  Additional keyword arguments are forwarded to the
        backend implementation.
    overwrite
        If *False* (default) and ``'cells'`` already exists in
        ``sdata.tables``, an exception is raised.  Set to *True* to recompute.
    return_table
        If *True*, only the detected *cells table* (:class:`pandas.DataFrame`)
        is returned instead of the full ``SpatialData``.
    **kwargs
        Extra keyword arguments forwarded verbatim to the backend detector
        (e.g. ``gpu=True``, ``batch_size=4``).

    Returns
    -------
    :class:`spatialdata.SpatialData` | :class:`pandas.DataFrame`
        Updated object containing a new ``'cells'`` table *or* the table alone
        if ``return_table`` is *True*.
    """

    # ------------------------------------------------------------------
    # 0) Normalise input into SpatialData object
    # ------------------------------------------------------------------
    # sdata = _ensure_spatialdata(sdata)

    logger.info("Detecting cells using backend '%s'…", model)

    # ------------------------------------------------------------------
    # 1) Dispatch to backend implementation (lazy import)
    # ------------------------------------------------------------------
    if model == "cellvit":
        from .._core.detection.cellvit import detect_cells_core as _backend
    # elif model == "stardist":
    #     from panospace._core.detection.stardist import detect_cells as _backend
    else:
        raise ValueError(f"Unknown detection model: {model!r}")

    cell_dict_wsi, cell_dict_detection = _backend(img, model_name='HIPT', device='cuda', tile_size=256, overlap=64)  # type: ignore[arg-type]

    logger.info("Detected %d cells", len(cell_dict_wsi))

    return cell_dict_wsi


__all__ = ["detect_cells"]
