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
    img
        High-resolution tissue image as a :class:`PIL.Image.Image`.  Large
        whole-slide images are automatically tiled before inference, so the
        input can be the full slide or an already cropped region.
    model
        Detection backend to use - currently ``"cellvit"`` (default).  Other
        identifiers (e.g. ``"stardist"``) are reserved for future
        implementations.

    Returns
    -------
    list[dict[str, Any]]
        The ``cell_dict_wsi`` output produced by the backend.  Each dictionary
        contains the detected cell's geometry and metadata in whole-slide
        coordinates (e.g. ``bbox``, ``centroid``, ``contour``, ``type`` and
        patch-level bookkeeping information).
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
