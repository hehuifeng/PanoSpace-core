"""CellViT backend for nuclei/cell detection, integrated with PanoSpace.

This version provides:
- minimal patch-wise CellViT inference
- automatic tiling for large WSI
- output in SpatialData with standard cell schema
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any, TYPE_CHECKING, Union, Literal
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd

from PIL import Image
Image.MAX_IMAGE_PIXELS = 10000000000

from panospace._core import register
# from panospace.io.adapters import _ensure_spatialdata  # type: ignore
# from panospace.io._schemas import SCHEMA_REGISTRY

if TYPE_CHECKING:
    from spatialdata import SpatialData  # for type hints

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Lazy import helper: minimal CellViT detector
# -----------------------------------------------------------------------------
def _lazy_detector():
    try:
        import torch
        from ._cellvit_backend.cellvit import (
            CellViT256, CellViTSAM
        )
        from ._cellvit_backend.cache_models import (
            cache_cellvit_256, cache_cellvit_sam
        )
        from ._cellvit_backend.tools import (
            unflatten_dict
        )
    except ImportError as e:
        raise ImportError(
            "CellViT backend needs PyTorch and CellViT code. "
            "Install via `pip install panospace[cellvit]`."
        ) from e
    import torch.nn.functional as F
    from torchvision import transforms

    class CellViTDetector:
        def __init__(self, model_name: Literal["HIPT", "SAM"], device: str = "cuda:0", resize=(256,256), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
            self.device = device
            self.model_name = model_name.upper()

            if self.model_name == "SAM":
                model_path = cache_cellvit_sam(logger=logger)  # 如果有 logger 可以传入
            elif self.model_name == "HIPT":
                model_path = cache_cellvit_256(logger=logger)
            else:
                raise ValueError("Unknown model name. Please use 'SAM' or 'HIPT'.")

            ckpt = torch.load(model_path, map_location="cpu")
            config = unflatten_dict(ckpt["config"], ".") if "config" in ckpt else ckpt["run_conf"]
            arch = ckpt["arch"]
            self.tranformer = transforms.Compose([
                transforms.Resize(resize),  
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

            if arch == "CellViT256":
                model = CellViT256(
                    model256_path=None,
                    num_nuclei_classes=config["data"]["num_nuclei_classes"],
                    num_tissue_classes=config["data"]["num_tissue_classes"],
                    regression_loss=config["model"].get("regression_loss", False),
                )
            elif arch == "CellViTSAM":
                model = CellViTSAM(
                    model_path=None,
                    num_nuclei_classes=config["data"]["num_nuclei_classes"],
                    num_tissue_classes=config["data"]["num_tissue_classes"],
                    vit_structure=config["model"]["backbone"],
                    regression_loss=config["model"].get("regression_loss", False),
                )
            else:
                raise NotImplementedError(f"Unsupported arch: {arch}")

            model.load_state_dict(ckpt["model_state_dict"])
            model.eval().to(self.device)
            self.model = model
            self.config = config
            self.arch = arch

        @torch.no_grad()
        def detect_patch(self, patch: Image.Image) -> pd.DataFrame:
            import torch

            # if patch.shape[-1] == 3:  # HWC → CHW
            #     patch = np.moveaxis(patch, -1, 0)
            # patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(self.device)
            patch = self.tranformer(patch).unsqueeze(0).to(self.device)
            with torch.no_grad():
                preds = self.model.forward(patch)

            preds = self.apply_softmax_reorder(preds)
            inst_maps, inst_types = self.model.calculate_instance_map(preds)

            return inst_maps, inst_types
        
        def apply_softmax_reorder(self, predictions: dict) -> dict:
            """Reorder and apply softmax on predictions

            Args:
                predictions(dict): Predictions

            Returns:
                dict: Predictions
            """
            predictions["nuclei_binary_map"] = F.softmax(
                predictions["nuclei_binary_map"], dim=1
            )
            predictions["nuclei_type_map"] = F.softmax(
                predictions["nuclei_type_map"], dim=1
            )
            predictions["nuclei_type_map"] = predictions["nuclei_type_map"].permute(
                0, 2, 3, 1
            )
            predictions["nuclei_binary_map"] = predictions["nuclei_binary_map"].permute(
                0, 2, 3, 1
            )
            predictions["hv_map"] = predictions["hv_map"].permute(0, 2, 3, 1)
            return predictions


    return CellViTDetector


# -----------------------------------------------------------------------------
# Simple slide tiler: for large images → patches with overlap
# -----------------------------------------------------------------------------
def _simple_tiler(img: np.ndarray, tile_size: int, overlap: int):
    """Yield (patch, x0, y0) for whole img."""
    H, W = img.shape[:2] if img.shape[-1] in (1, 3, 4) else img.shape[1:]
    stride = tile_size - overlap
    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            patch = img[y0:y0 + tile_size, x0:x0 + tile_size]
            pad_bottom = max(0, tile_size - patch.shape[0])
            pad_right = max(0, tile_size - patch.shape[1])
            if pad_bottom > 0 or pad_right > 0:
                patch = cv2.copyMakeBorder(patch, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            yield patch, int(x0), int(y0), int(1 + x0 // stride), int(1 + y0 // stride)


def _simple_tiler_pil(img: Image.Image, tile_size: int, overlap: int):
    W, H = img.size  # 注意 PIL 是 (宽, 高)
    stride = tile_size - overlap

    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            x1 = min(x0 + tile_size, W)
            y1 = min(y0 + tile_size, H)
            tile = img.crop((x0, y0, x1, y1))

            # 补白（右下角超出图像边界的情况）
            if tile.size != (tile_size, tile_size):
                padded = Image.new(img.mode, (tile_size, tile_size), color=(255, 255, 255))
                padded.paste(tile, (0, 0))
                tile = padded

            yield tile, x0, y0, int(1 + x0 // stride), int(1 + y0 // stride)


# -----------------------------------------------------------------------------
# Public API: main entry
# -----------------------------------------------------------------------------
def detect_cells_core(
    img: Image.Image,
    *,
    model_name: Literal["HIPT", "SAM"],
    device: str = "cuda:0",
    tile_size: int = 256,
    overlap: int = 64,
):
    """Run CellViT nuclei detection on a raw image array.

    Automatically tiles large images patch-wise and merges results.

    Parameters
    ----------
    img : Image.Image
        Raw image as a Image.Image.
    model_weights : str or Path
        Path to a `.pth` or `.pt` checkpoint.
    device : str
        Torch device string (e.g., 'cuda:0' or 'cpu').
    tile_size : int
        Size of tiles for inference.
    overlap : int
        Overlap between tiles.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ["cell_id", "x", "y", "morphotype"]
    """
    t0 = perf_counter()

    # if img.ndim != 3:
    #     raise ValueError("Expected image with 3 dimensions (HWC or CHW).")
    # if img.shape[0] in {3, 4}:
    #     img = np.moveaxis(img, 0, -1)  # Convert CHW -> HWC

    CellViTDetector = _lazy_detector()
    detector = CellViTDetector(model_name, device)

    logger.info(
        "[CellViT] Running inference on image shape %s (tile %d, overlap %d)",
        img.size, tile_size, overlap,
    )

    dfs = []
    # print('img size:', img.shape)
    cell_dict_wsi=[]
    cell_dict_detection=[]  
    from ._cellvit_backend.postprocessing import process_cell_instance

    # 预先计算分块总数
    W, H = img.size
    stride = tile_size - overlap
    n_cols = int(np.ceil((W - tile_size) / stride)) + 1
    n_rows = int(np.ceil((H - tile_size) / stride)) + 1
    total_tiles = n_cols * n_rows

    tiles_generator = _simple_tiler_pil(img, tile_size, overlap)
    for patch, x0, y0, col, row in tqdm(tiles_generator, total=total_tiles):
        offset_global=np.array([y0,x0])

        _, inst_types = detector.detect_patch(patch)
    
        # return inst_types
        cell_dict,cell_detection=process_cell_instance(instance_types=inst_types[0], offset_global=offset_global,
                                                        row=row, col=col, tile_size=tile_size, overlap=overlap)
        cell_dict_wsi.extend(cell_dict)
        cell_dict_detection.extend(cell_detection)
    
    from ._cellvit_backend.postprocessing import CellPostProcessor
    cell_processor = CellPostProcessor(cell_list=cell_dict_wsi)
    cleaned_cells= cell_processor.post_process_cells()
    keep_idx=list(cleaned_cells.index.values)
    cell_dict_wsi = [cell_dict_wsi[idx_c] for idx_c in keep_idx]
    cell_dict_detection = [cell_dict_detection[idx_c] for idx_c in keep_idx]

    return cell_dict_wsi, cell_dict_detection

# -----------------------------------------------------------------------------
# Register to PanoSpace detection backend
# -----------------------------------------------------------------------------
register("detection", "cellvit", detect_cells_core)
