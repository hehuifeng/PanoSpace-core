"""panospace.io.adapters
=======================
I/O *adapter* functions that translate various vendor-specific file layouts into
**SpatialData** objects, and vice versa.

Design goals
------------
*   **Thin wrappers** around `spatialdata_io.*` helpers - we add uniform argument
    names, logging, error handling and post-processing so that downstream
    PanoSpace code can rely on consistent `SpatialData` annotations.
*   **Lazy optional deps** - heavy packages like `scanpy`, `spatialdata_io`,
    `h5py`, etc. are imported *within* each function, so importing
    ``panospace.io`` stays lightweight.
*   **Schema normalisation** - after reading, we add/rename standard columns
    (e.g. `obs['x']`, `obs['y']`, `obs['z']`), and ensure a valid Common
    Coordinate System (CCS) exists.
"""
from __future__ import annotations

from pathlib import Path
import logging
from typing import Any, Mapping

# import spatialdata as sdata  # lightweight; no heavy IO backends here

# import xarray
# import dask

# logger = logging.getLogger(__name__)

# # -----------------------------------------------------------------------------
# # Helpers
# # -----------------------------------------------------------------------------

# def _as_path(path: str | Path) -> Path:
#     """Return *absolute* :class:`~pathlib.Path` and raise if missing."""
#     p = Path(path).expanduser().resolve()
#     if not p.exists():
#         raise FileNotFoundError(f"File or directory not found: {p}")
#     return p


# def _postprocess(spatial: "sdata.SpatialData", *, source: str) -> "sdata.SpatialData":
#     """Ensure mandatory metadata & log the result."""
#     if "spots" in spatial.tables:
#         tbl = spatial.tables["spots"]
#         if {"x", "y"}.issubset(tbl.columns):
#             tbl.spatial_key = "{x,y}"
#     logger.info("[%s] Loaded SpatialData: images=%d, tables=%d", source, len(spatial.images), len(spatial.tables))
#     return spatial

# # -----------------------------------------------------------------------------
# # Vendor readers
# # -----------------------------------------------------------------------------

# def read_visium(folder: str | Path, *, image_path: str | Path | None = None, **kwargs: Any) -> "sdata.SpatialData":
#     """Read **10x Visium** output directory and return a :class:`SpatialData`.

#     Parameters
#     ----------
#     folder
#         Path to the *Visium* spaceranger output (should contain `filtered_feature_bc_matrix.h5`).
#     image_path
#         Optional path to the high-resolution tissue image; if *None*, tries to
#         find `tissue_hires_image.png` or `.jpg` automatically.
#     kwargs
#         Additional keyword arguments forwarded to
#         :func:`spatialdata_io.read_visium`.
#     """
#     from spatialdata_io import visium  # local import (heavy)

#     folder = _as_path(folder)
#     if image_path is not None:
#         image_path = _as_path(image_path)

#     sd = visium(path=folder, image_path=image_path, **kwargs)
#     return _postprocess(sd, source="Visium")


# def read_xenium(csv_or_folder: str | Path, **kwargs: Any) -> "sdata.SpatialData":
#     """Read **10x Xenium** CSV folder or zipped file into :class:`SpatialData`."""
#     from spatialdata_io import xenium

#     path = _as_path(csv_or_folder)
#     sd = xenium(path=path, **kwargs)
#     return _postprocess(sd, source="Xenium")


# def read_cosmx(folder: str | Path, **kwargs: Any) -> "sdata.SpatialData":
#     """Read **NanoString CosMx** data into :class:`SpatialData`."""
#     from spatialdata_io import cosmx

#     folder = _as_path(folder)
#     sd = cosmx(path=folder, **kwargs)
#     return _postprocess(sd, source="CosMx")


# def read_stereo(folder: str | Path, **kwargs: Any) -> "sdata.SpatialData":
#     """Read **BGI Stereo-seq** GEM/JSON output into :class:`SpatialData`."""
#     from spatialdata_io import read_stereo

#     folder = _as_path(folder)
#     sd = read_stereo(path=folder, **kwargs)
#     return _postprocess(sd, source="Stereo-seq")

# # -----------------------------------------------------------------------------
# # Converters
# # -----------------------------------------------------------------------------

# def to_anndata(spatial: "sdata.SpatialData", *, table_key: str = "cells", layer: str | None = None):
#     """Extract an :class:`~anndata.AnnData` view from a SpatialData *table*.

#     Parameters
#     ----------
#     spatial
#         Input SpatialData.
#     table_key
#         Key of the table (e.g. "cells" or "spots").
#     layer
#         Layer name to move to ``.X``; if *None*, keeps default.
#     """
#     import anndata as ad

#     tbl = spatial.tables[table_key]
#     adata = tbl.to_anndata()
#     if layer is not None and layer in adata.layers:
#         adata.X = adata.layers[layer]
#     return adata


# def to_spatialdata(adata: "Any", *, img: str | Path | None = None, **kwargs: Mapping[str, Any]):
#     """Wrap an :class:`~anndata.AnnData` into **SpatialData** with optional image."""
#     import anndata as ad

#     if not isinstance(adata, ad.AnnData):
#         raise TypeError("`adata` must be an AnnData object.")
#     from spatialdata import SpatialData, Table

#     tbl = Table.from_anndata(adata, table_name="cells")
#     imgs = {}
#     if img is not None:
#         from spatialdata import Image2D
#         img_p = _as_path(img)
#         imgs["image"] = Image2D.from_file(img_p, **kwargs)

#     sd = SpatialData(images=imgs, tables={"cells": tbl})  # shapes, points left empty
#     return sd


# def from_adata_and_image(
#     adata: "Any",
#     image_path: str | Path,
#     *,
#     image_name: str = "image",
#     image_channel_dim: str = "c",
#     check_coords: bool = True,
# ) -> "sdata.SpatialData":
#     """
#     Create a SpatialData object from an AnnData and a single image file.

#     Parameters
#     ----------
#     adata : AnnData
#         The input expression matrix with spatial metadata.
#     image_path : str or Path
#         Path to a single image (.tif, .png, .jpg).
#     image_name : str, default "image"
#         The name to store the image under in SpatialData.
#     image_channel_dim : str, default "c"
#         Name of the image channel axis (typically "c").
#     check_coords : bool, default True
#         If True, ensures that AnnData contains spatial coordinates.

#     Returns
#     -------
#     SpatialData
#         A complete SpatialData object with image + table.
#     """
#     import anndata as ad
#     import xarray as xr
#     import dask.array as da
#     from tifffile import imread
#     from PIL import Image
#     import numpy as np
#     from spatialdata import SpatialData
#     from spatialdata.models import TableModel
#     from pathlib import Path as _as_path  # 若 _as_path 未定义，显式加上

#     if not isinstance(adata, ad.AnnData):
#         raise TypeError("`adata` must be an AnnData object.")

#     if check_coords:
#         has_obsm = "spatial" in adata.obsm
#         has_obs_xy = {"x", "y"}.issubset(adata.obs.columns)
#         if not (has_obsm or has_obs_xy):
#             raise ValueError(
#                 "AnnData must have either `.obsm['spatial']` or ['x', 'y'] columns in `.obs` "
#                 "to define spatial coordinates."
#             )

#     # 自动添加 SpatialData 所需的 region_key 和 instance_key
#     if "fov" not in adata.obs.columns:
#         adata.obs["fov"] = "global"
#     if "cell_ID" not in adata.obs.columns:
#         adata.obs["cell_ID"] = adata.obs_names

#     tbl = TableModel.parse(
#         adata,
#         region="global",
#         region_key="fov",
#         instance_key="cell_ID"
#     )

#     # 读取图像
#     p = _as_path(image_path)
#     if p.suffix.lower() in [".tif", ".tiff"]:
#         img = imread(str(p))
#     elif p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
#         img = np.asarray(Image.open(str(p)))
#     else:
#         raise ValueError(f"Unsupported image format: {p.suffix}")

#     # 标准化图像维度
#     if img.ndim == 2:
#         img = img[None, :, :]
#     elif img.ndim == 3:
#         if img.shape[2] in [3, 4]:
#             img = img.transpose(2, 0, 1)
#         elif img.shape[0] in [3, 4]:
#             pass
#         else:
#             raise ValueError(f"Ambiguous image shape: {img.shape}")
#     else:
#         raise ValueError(f"Unsupported image ndim: {img.ndim}")
    
#     img = da.from_array(img)

#     da_xr = xr.DataArray(img, dims=(image_channel_dim, "y", "x"))
#     da_xr.attrs["transform"] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

#     return SpatialData(images={image_name: da_xr}, tables={"cells": tbl})



# # -----------------------------------------------------------------------------
# # Public re-exports
# # -----------------------------------------------------------------------------
# __all__ = [
#     "read_visium",
#     "read_xenium",
#     "read_cosmx",
#     "read_stereo",
#     "to_anndata",
#     "to_spatialdata",
#     "from_adata_and_image",
# ]
