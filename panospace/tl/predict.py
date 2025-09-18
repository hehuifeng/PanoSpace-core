"""panospace.tl.predict
=====================

"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

from . import _import_backend

logger = logging.getLogger("panospace.tl")

# -----------------------------------------------------------------------------
# Backend registry ----------------------------------------------------------------
# -----------------------------------------------------------------------------
def genexp_predictor(
    sc_adata: AnnData,
    spot_adata: AnnData,
    infered_adata: AnnData,
    celltype_list: list[str],
    celltype_column: str = "celltype_major",
    backend: str = "predictor",
) -> AnnData:
    """
    Predict spatial gene expression profiles from single-cell references.

    This function leverages scRNA-seq data as a reference and integrates
    spatial deconvolution results to reconstruct gene expression patterns
    across spatial transcriptomics spots or subspot-resolution locations.

    Workflow
    --------
    1. Validate input objects: reference scRNA (`sc_adata`), spatial data (`spot_adata`),
       and inferred spatial assignments (`infered_adata`).
    2. Select the prediction backend via :func:`_import_backend`.
    3. Estimate cell-type-specific expression contributions and reconstruct
       predicted gene expression matrix for each spatial location.
    4. Return an AnnData object with the predicted expression.

    Parameters
    ----------
    sc_adata : AnnData
        Reference single-cell RNA-seq dataset with cell-type annotations.
    spot_adata : AnnData
        Spatial transcriptomics dataset (spot-level expression).
    infered_adata : AnnData
        Inferred spatial decomposition results (e.g. cell-type fractions).
    celltype_list : list of str
        List of cell types to include in the prediction model.
    celltype_column : str, default="celltype_major"
        Column name in `sc_adata.obs` that stores cell-type labels.
    backend : {"predictor"}, default="predictor"
        Backend implementation for prediction. Currently only supports `"predictor"`.

    Returns
    -------
    AnnData
        AnnData object containing the reconstructed spatial gene expression
        profiles. The structure typically mirrors `spot_adata`, but with
        predicted expression values.

    Raises
    ------
    ValueError
        If the specified backend is not supported or fails to initialize.

    Example
    -------
    >>> import panospace as ps
    >>> sc_ref = ps.io.to_anndata(ps.io.read_xenium("sample/xenium"))
    >>> visium_data = ps.io.read_visium("sample/visium")
    >>> inferred = ps.tl.annotate(sc_ref, visium_data, backend="RCTD")
    >>> pred = ps.tl.predict.genexp_predictor(
    ...     sc_adata=sc_ref,
    ...     spot_adata=visium_data,
    ...     infered_adata=inferred,
    ...     celltype_list=["Astrocyte", "Neuron", "Oligodendrocyte"]
    ... )
    """

    start_time = time.time()
    logger.info(f"Starting gene expression prediction using backend '{backend}'")

    # 动态加载后端实现函数（例如 predictor.py）
    backend_func = _import_backend(backend)

    # 调用具体后端执行预测
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