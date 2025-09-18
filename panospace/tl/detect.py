"""panospace.tl.detect
====================
User-facing *nuclei/cell detection* wrapper.  It delegates the heavy lifting to
backend implementations in :pymod:`panospace._core.detection` (CellViT,
StarDist, ...).

"""

from __future__ import annotations

import logging
from typing import Any, Literal

from PIL import Image
Image.MAX_IMAGE_PIXELS = 10000000000

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

    seg_adata, contours = _backend(img, model_name='HIPT', device='cuda', tile_size=256, overlap=64)  # type: ignore[arg-type]

    logger.info("Detected %d cells", len(seg_adata))

    return seg_adata, contours


__all__ = ["detect_cells"]
