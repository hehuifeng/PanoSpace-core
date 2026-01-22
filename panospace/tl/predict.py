"""panospace.tl.predict
=====================
Gene expression prediction for spatial transcriptomics data.

This module provides tools to predict spatial gene expression profiles
from single-cell RNA-seq references, leveraging deconvolution results.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

from . import _import_backend

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def genexp_predictor(
    sc_adata: AnnData,
    spot_adata: AnnData,
    infered_adata: AnnData,
    celltype_list: list[str],
    celltype_column: str = "celltype_major",
    backend: str = "predictor",
) -> AnnData:
    """Predict spatial gene expression from single-cell reference.

    Reconstructs gene expression patterns across spatial locations by
    integrating scRNA-seq reference with deconvolution results.

    Parameters
    ----------
    sc_adata : AnnData
        Reference single-cell RNA-seq data with cell-type annotations.
    spot_adata : AnnData
        Spatial transcriptomics data (spot-level expression).
    infered_adata : AnnData
        Spatial decomposition results (e.g., cell-type fractions).
    celltype_list : list of str
        Cell types to include in prediction.
    celltype_column : str, default="celltype_major"
        Column in `sc_adata.obs` storing cell-type labels.
    backend : {"predictor"}, default="predictor"
        Prediction backend. Currently only `"predictor"` is supported.

    Returns
    -------
    AnnData
        Spatial gene expression predictions. Structure mirrors `spot_adata`
        with predicted expression values.

    Examples
    --------
    >>> import panospace as ps
    >>> # After deconvolution
    >>> pred = ps.genexp_predictor(
    ...     sc_adata=sc_ref,
    ...     spot_adata=visium_data,
    ...     infered_adata=deconv_result,
    ...     celltype_list=["Astrocyte", "Neuron", "Oligodendrocyte"]
    ... )
    """

    start_time = time.time()
    logger.info(f"Starting gene expression prediction using backend '{backend}'")

    # Dynamically load backend
    backend_func = _import_backend(backend)

    # Execute prediction
    adata_pred = backend_func(
        sc_adata=sc_adata,
        spot_adata=spot_adata,
        infered_adata=infered_adata,
        celltype_list=celltype_list,
        celltype_column=celltype_column,
    )

    logger.info(
        f"Gene expression prediction completed in {time.time() - start_time:.2f} seconds."
    )
    return adata_pred