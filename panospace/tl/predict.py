"""panospace.tl.predict
=====================
High‑level wrapper for **single‑cell gene‑expression prediction** (sometimes
referred to as *super‑resolution* when the input is spot‑level data).

The function assumes that you have already

1. **Loaded** a low‑resolution spatial‑transcriptomics dataset as a
   :class:`spatialdata.SpatialData` object, *or* passed a path that can be read
   by :pymod:`panospace.io.adapters`.
2. **Detected cells** – the ``cells`` table must exist (created by
   :pyfunc:`panospace.tl.detect_cells`).
3. **Annotated cell types** – column ``obs['cell_type']`` is required (created
   by :pyfunc:`panospace.tl.annotate_celltype`).

It then calls a backend algorithm (default: graph‑based semi‑supervised
learning) that outputs a dense **cell × gene** expression matrix.  The matrix is
written into ``sdata.layers[layer_name]`` and the updated object is returned so
that the call can be chained.

Parameters
----------
 sdata
     A :class:`spatialdata.SpatialData` object *or* a path / directory that can
     be parsed into one.  The object **must** contain a valid ``cells`` table
     with at least the columns ``cell_id`` and ``cell_type``.
 scrna_ref
     The matched single‑cell reference used for training.  Accepts an
     :class:`anndata.AnnData`, a path to ``.h5ad`` or a ``SpatialData`` whose
     ``cells`` table has the reference expression.  Only genes present in both
     the reference and the spatial dataset will be used.
 layer_name
     Name of the ``SpatialData.layers`` slot where the predicted expression
     matrix will be stored.  Defaults to ``"expr_pred"``.
 backend
     Identifier of the backend implementation.  Currently supports
     ``"graph_ssl"`` (default) and experimental ``"gnn_superres"``.  Plugins can
     register additional backends via Python *entry‑points*.
 overwrite
     If *False* (default) and the target `layer_name` already exists, raise a
     :class:`ValueError`.  Set to *True* to overwrite.
 select_genes
     ``"all"`` (default) uses the union of genes present in both datasets.
     ``"highly_variable"`` will restrict to reference genes flagged as highly
     variable (``scrna_ref.var['highly_variable']``).  Alternatively pass a
     ``Sequence[str]`` of gene names to use.
 return_array
     If *True*, the function returns a tuple ``(sdata, expr_matrix)`` where
     *expr_matrix* is the predicted ``np.ndarray`` (or ``scipy.sparse``).  If
     *False*, only the updated ``SpatialData`` is returned.
 **backend_kwargs
     Additional keyword arguments forwarded to the selected backend.

Returns
-------
 SpatialData
     The same object that was passed in, updated in‑place.
 ndarray or sparse matrix, *optional*
     Only returned if *return_array* is *True*.

Examples
--------
>>> import panospace as ps
>>> sdata = ps.io.read_visium("sample/visium")
>>> sdata = ps.tl.detect_cells(sdata)
>>> sdata = ps.tl.annotate_celltype(sdata, scrna_ref="atlas.h5ad")
>>> sdata = ps.tl.predict_expr(sdata, scrna_ref="atlas.h5ad")
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Mapping, Sequence, Tuple, Union, overload

import numpy as np
import pandas as pd

try:
    import spatialdata as sdata_mod  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("spatialdata is required for panospace.tl.predict") from e

try:
    import anndata as ad
except ImportError as e:  # pragma: no cover
    raise ImportError("anndata is required for panospace.tl.predict") from e

from ..io.adapters import _ensure_spatialdata
from ..io._schemas import SCHEMA_REGISTRY
from .._utils.logging import timer

logger = logging.getLogger("panospace.tl")

# -----------------------------------------------------------------------------
# Backend registry ----------------------------------------------------------------
# -----------------------------------------------------------------------------

_BACKENDS: Mapping[str, str] = {
    "graph_ssl": "panospace._core.prediction.graph_ssl:predict_expr_core",
    "gnn_superres": "panospace._core.prediction.tesla_gnn:predict_expr_core",
}


# -----------------------------------------------------------------------------
# Public API -------------------------------------------------------------------
# -----------------------------------------------------------------------------

@overload
def predict_expr(
    sdata: Union[str, Path, "sdata_mod.SpatialData"],
    /,
    *,
    scrna_ref: Union[str, Path, "ad.AnnData", "sdata_mod.SpatialData"],
    layer_name: str = "expr_pred",
    backend: str = "graph_ssl",
    overwrite: bool = False,
    select_genes: Union[str, Sequence[str]] = "all",
    return_array: bool = False,
    **backend_kwargs,
) -> "sdata_mod.SpatialData":
    ...


@overload
def predict_expr(
    sdata: Union[str, Path, "sdata_mod.SpatialData"],
    /,
    *,
    scrna_ref: Union[str, Path, "ad.AnnData", "sdata_mod.SpatialData"],
    layer_name: str = "expr_pred",
    backend: str = "graph_ssl",
    overwrite: bool = False,
    select_genes: Union[str, Sequence[str]] = "all",
    return_array: bool = True,
    **backend_kwargs,
) -> Tuple["sdata_mod.SpatialData", "np.ndarray"]:
    ...


def predict_expr(
    sdata: Union[str, Path, "sdata_mod.SpatialData"],
    /,
    *,
    scrna_ref: Union[str, Path, "ad.AnnData", "sdata_mod.SpatialData"],
    layer_name: str = "expr_pred",
    backend: str = "graph_ssl",
    overwrite: bool = False,
    select_genes: Union[str, Sequence[str]] = "all",
    return_array: bool = False,
    **backend_kwargs,
):  # noqa: D401 – Google vs NumPy doc clash
    """Predict gene expression for every detected cell."""

    t0 = time.perf_counter()
    sdata = _ensure_spatialdata(sdata)

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    if "cells" not in sdata.tables:
        raise ValueError("`sdata` has no `cells` table – run tl.detect_cells() first.")

    cells_df = sdata.tables["cells"].to_pandas()
    SCHEMA_REGISTRY["cells"].validate(cells_df)

    if "cell_type" not in cells_df.columns:
        raise ValueError("`cells.obs['cell_type']` missing – run tl.annotate_celltype().")

    if not overwrite and layer_name in sdata.layers:
        raise ValueError(
            f"Layer '{layer_name}' already exists. Use `overwrite=True` to replace.")

    # ------------------------------------------------------------------
    # Prepare reference AnnData
    # ------------------------------------------------------------------
    if isinstance(scrna_ref, (str, Path)):
        scrna_ref = ad.read_h5ad(scrna_ref)  # type: ignore[arg-type]
    elif isinstance(scrna_ref, sdata_mod.SpatialData):
        scrna_ref = ad.AnnData(scrna_ref.tables["cells"].to_pandas())
    elif not isinstance(scrna_ref, ad.AnnData):  # type: ignore[misc]
        raise TypeError("`scrna_ref` must be AnnData, path to .h5ad or SpatialData.")

    # Gene selection
    if isinstance(select_genes, str):
        if select_genes == "all":
            genes = scrna_ref.var_names.intersection(sdata.var_names)  # type: ignore[attr-defined]
        elif select_genes == "highly_variable":
            if "highly_variable" not in scrna_ref.var.columns:
                raise ValueError("`scrna_ref.var['highly_variable']` not found.")
            hv = scrna_ref.var.index[scrna_ref.var["highly_variable"]]
            genes = hv.intersection(sdata.var_names)  # type: ignore[attr-defined]
        else:
            raise ValueError("`select_genes` must be 'all', 'highly_variable' or list of genes.")
    else:
        genes = pd.Index(select_genes)

    # ------------------------------------------------------------------
    # Resolve backend & run prediction
    # ------------------------------------------------------------------
    if backend not in _BACKENDS:
        raise ValueError(f"Backend '{backend}' is not registered.")

    module_name, func_name = _BACKENDS[backend].split(":")
    backend_fn = __import__(module_name, fromlist=[func_name]).__dict__[func_name]

    logger.info("[predict_expr] backend=%s | genes=%d | cells=%d", backend, len(genes), len(cells_df))
    with timer("predict_expr", logger):
        expr_mat = backend_fn(
            sdata=sdata,
            scrna_ref=scrna_ref,
            genes=genes,
            **backend_kwargs,
        )

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    if expr_mat.shape != (len(cells_df), len(genes)):
        raise ValueError("Backend returned matrix with incompatible shape.")

    sdata.layers[layer_name] = expr_mat
    sdata.uns.setdefault("panospace", {})["predict"] = {
        "backend": backend,
        "n_genes": len(genes),
        "layer": layer_name,
        "runtime_sec": round(time.perf_counter() - t0, 2),
    }

    logger.info("[predict_expr] finished in %.2f s", time.perf_counter() - t0)

    if return_array:
        return sdata, expr_mat
    return sdata


__all__ = ["predict_expr"]
