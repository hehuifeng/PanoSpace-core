"""panospace.tl.annotate
======================
High-level wrapper for **cell-type annotation** and deconvolution at single-cell resolution.

This module provides a unified interface to:
    1. Load and validate a reference scRNA-seq dataset.
    2. Run selected backend algorithms to map reference cell-type labels
       onto detected cells in a spatial transcriptomics dataset.
    3. Write results back to the target AnnData object (e.g., probability
       matrices, cell-type assignments).

Backends
--------
Supported backends for gene-expression-based cell-type deconvolution:
* ``"RCTD"``         - Robust Cell Type Decomposition.
* ``"cell2location"`` - Probabilistic cell-type mapping model.
* ``"spatialDWLS"``   - Deconvolution via dampened weighted least squares.
* ``"endecon"``       - Ensemble integration of multiple backend results.

Example
-------
>>> import panospace as ps
>>> sdata = ps.io.read_visium("sample/visium")
>>> ref_ad = ps.io.to_anndata(ps.io.read_xenium("sample/xenium"))
>>> sdata = ps.tl.detect_cells(sdata)
>>> sdata = ps.tl.deconv_celltype(sdata, ref_ad, celltype_key="cell_type")
"""

from __future__ import annotations

import logging
from typing import List, Literal, Mapping

import anndata as ad
import pandas as pd
from anndata import AnnData

logger = logging.getLogger("panospace.tl")

# -----------------------------------------------------------------------------
# Backend registry
# -----------------------------------------------------------------------------

_BACKENDS: Mapping[str, str] = {
    "RCTD": "panospace._core.annotation.RCTD:annotate_cells_core",
    "cell2location": "panospace._core.annotation.cell2location:annotate_cells_core",
    "spatialDWLS": "panospace._core.annotation.spatialDWLS:annotate_cells_core",
    "endecon": "panospace._core.annotation.endecon:endecon_core",
}


# -----------------------------------------------------------------------------
# Helper: Lazy backend import
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def deconv_celltype(
    adata_vis: AnnData,
    sc_adata: AnnData,
    celltype_key: str,
    methods: List[Literal["RCTD", "cell2location", "spatialDWLS"]] = ["RCTD", "cell2location", "spatialDWLS"],
) -> AnnData:
    """
    Perform cell-type deconvolution using multiple backends and integrate results via EnDecon.

    This function:
        1. Runs multiple deconvolution backends (e.g., RCTD, cell2location, spatialDWLS).
        2. Aligns and ensembles their results using the EnDecon method.
        3. Stores individual and ensemble outputs into `adata_vis.uns`.

    Parameters
    ----------
    adata_vis : AnnData
        Spatial transcriptomics AnnData object (e.g., Visium data) containing
        expression counts of spots or detected cells.
    sc_adata : AnnData
        Reference single-cell RNA-seq AnnData object with known cell-type annotations.
    celltype_key : str
        Column key in `sc_adata.obs` specifying cell-type labels.
    methods : list of {"RCTD", "cell2location", "spatialDWLS"}, optional
        List of backends to run for deconvolution. Default includes all three.

    Returns
    -------
    AnnData
        The input `adata_vis` with additional results stored in:
        - `adata_vis.uns["X_deconv_<backend>"]`: raw results from each backend.
        - `adata_vis.uns["X_deconv_ensemble"]`: ensemble integration result,
          with probability matrix stored in `["H_norm"]`.

    Raises
    ------
    KeyError
        If `celltype_key` is not found in `sc_adata.obs`.
    """
    # Ensure the reference dataset contains cell-type annotations
    if celltype_key not in sc_adata.obs:
        raise KeyError(f"Reference AnnData must contain '{celltype_key}' in .obs")

    results_list = []

    # Run each selected backend
    for method in methods:
        logger.info("Running cell-type deconvolution using backend '%s'", method)
        backend_fn = _import_backend(method)

        result_df: pd.DataFrame = backend_fn(
            sc_adata=sc_adata,
            adata_vis=adata_vis,
            celltype_key=celltype_key,
        )

        # Store per-backend result in `uns`
        adata_vis.uns[f"X_deconv_{method}"] = result_df
        results_list.append(result_df)

    # Determine common genes (columns) and common cells (rows) across all results
    common_index = sorted(set.intersection(*(set(df.index) for df in results_list)))
    common_columns = sorted(set.intersection(*(set(df.columns) for df in results_list)))

    # Align results to the same index/columns and convert to NumPy arrays
    aligned_results = [
        df.loc[common_index, common_columns].to_numpy(dtype=float) for df in results_list
    ]

    # Ensemble integration
    logger.info("Running EnDecon ensemble integration...")
    EnDecon = _import_backend("endecon")
    ensemble_result = EnDecon(aligned_results)

    # Convert H_norm (probability matrix) to DataFrame
    ensemble_result["H_norm"] = pd.DataFrame(
        ensemble_result["H_norm"], index=common_index, columns=common_columns
    )

    # Save ensemble result
    adata_vis.uns["X_deconv_ensemble"] = ensemble_result

    return adata_vis


__all__ = ["deconv_celltype"]