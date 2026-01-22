"""panospace.tl.detect
====================
User-facing *nuclei/cell detection* wrapper.  It delegates the heavy lifting to
backend implementations in :pymod:`panospace._core.detection` (CellViT,
StarDist, ...).

"""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from PIL import Image
Image.MAX_IMAGE_PIXELS = 10000000000

logger = logging.getLogger(__name__)

from .._utils.device_utils import get_device


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def detect_cells(
    img: Image.Image,
    *,
    model: Literal["cellvit", "stardist"] = "cellvit",
    model_name: str = "HIPT",
    device: Optional[Literal["cuda", "cpu"]] = None,
    tile_size: Optional[int] = None,
    overlap: int = 64,
    prefer_gpu: bool = True,
    **kwargs: Any,
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
    model_name
        Name of the model to use within the selected backend. Default is "HIPT".
        Supported models: "HIPT", "SAM" (CellViT variants).
    device
        Device to use for inference ("cuda", "cpu", or None for auto-detection).
        If None, will automatically detect the best available device.
    tile_size
        Size of tiles for processing large images. CellViT models expect inputs
        that are multiples of their patch size (typically 256). Recommended values:
        256, 512, 768, 1024, etc. If None, defaults to 256.
        WARNING: CellViT models resize tiles internally, so any tile_size is
        technically supported, but values that are multiples of 256 are most efficient.
    overlap
        Overlap between tiles in pixels to avoid edge artifacts. Default is 64.
        Recommended to be at least 32 and less than tile_size/4.
    prefer_gpu
        Whether to prefer GPU when auto-detecting device. Ignored if device is explicitly set.
    **kwargs
        Additional keyword arguments passed to the backend.

    Returns
    -------
    tuple
        A tuple containing:
        - seg_adata: AnnData object with segmentation results
        - contours: Cell contours and metadata

    Examples
    --------
    >>> from PIL import Image
    >>> import panospace as ps
    >>> # Load image from file
    >>> img = Image.open("tissue.tif")
    >>> # Basic usage - tile_size=256 is recommended for CellViT models
    >>> seg_adata, contours = ps.detect_cells(img, tile_size=256)
    >>> # For larger images, use larger tiles (must be multiple of 256 for efficiency)
    >>> seg_adata, contours = ps.detect_cells(img, tile_size=512)
    >>> # Force CPU inference
    >>> seg_adata, contours = ps.detect_cells(img, device="cpu", tile_size=256)
    >>> # With smaller overlap for faster processing
    >>> seg_adata, contours = ps.detect_cells(img, tile_size=256, overlap=32)

    Notes
    -----
    CellViT models are trained on 256x256 patches. While any tile_size will work
    (images are resized internally), using tile sizes that are multiples of 256
    (e.g., 256, 512, 768) is more efficient as it reduces the number of resize operations.

    For very large images (>500MB), consider using:
    - tile_size=256 for most compatibility
    - tile_size=512 or 768 for faster processing (if memory permits)
    - Overlap of 64-128 pixels to avoid edge artifacts between tiles.
    """
    # ------------------------------------------------------------------
    # 0) Device detection and parameter setup
    # ------------------------------------------------------------------
    if device is None:
        device = get_device(prefer_gpu=prefer_gpu)

    # Validate device
    if device not in ["cuda", "cpu"]:
        raise ValueError(f"Device must be 'cuda' or 'cpu', got {device!r}")

    # Set tile_size (explicit user specification is recommended)
    if tile_size is None:
        tile_size = 256  # Default fallback
        logger.info("No tile_size specified, using default: tile_size=256. "
                   "For better performance, specify tile_size explicitly (256, 512, 768, etc.)")

    logger.info("Detecting cells using backend '%s' with device='%s', tile_size=%d",
                model, device, tile_size)

    # ------------------------------------------------------------------
    # 1) Dispatch to backend implementation (lazy import)
    # ------------------------------------------------------------------
    if model == "cellvit":
        from .._core.detection.cellvit import detect_cells_core as _backend
    # elif model == "stardist":
    #     from panospace._core.detection.stardist import detect_cells as _backend
    else:
        raise ValueError(f"Unknown detection model: {model!r}")

    # Call backend with configurable parameters
    seg_adata, contours = _backend(
        img,
        model_name=model_name,
        device=device,
        tile_size=tile_size,
        overlap=overlap,
        **kwargs
    )

    logger.info("Detected %d cells", len(seg_adata))

    return seg_adata, contours


__all__ = ["detect_cells"]
