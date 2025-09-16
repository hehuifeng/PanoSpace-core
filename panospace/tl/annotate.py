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

import pandas as pd
from anndata import AnnData

logger = logging.getLogger("panospace.tl")

from tl import _import_backend

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
    adata_vis.uns['cell_type'] = celltype_key

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
    # common_columns = sorted(set.intersection(*(set(df.columns) for df in results_list)))

    # Align results to the same index/columns and convert to NumPy arrays
    aligned_results = [
        df.loc[common_index, celltype_key].to_numpy(dtype=float) for df in results_list
    ]

    # Ensemble integration
    logger.info("Running EnDecon ensemble integration...")
    EnDecon = _import_backend("endecon")
    ensemble_result = EnDecon(aligned_results)

    # Convert H_norm (probability matrix) to DataFrame
    ensemble_result["H_norm"] = pd.DataFrame(
        ensemble_result["H_norm"], index=common_index, columns=celltype_key
    )

    # Save ensemble result
    deconv_adata = adata_vis.copy()
    deconv_adata = deconv_adata[common_index].copy()
    deconv_adata.obs = deconv_adata.obs.join(ensemble_result["H_norm"], how="left")

    return deconv_adata

def superres_celltype(
    adata_vis: AnnData,
    deconv_adata: AnnData,
    img_dir: str,
    neighb: int=3,
    radius: int=129,
    epoch: int=50,
    batch_size: int=32,
    num_workers: int=4,
    accelerator: Literal['cpu', 'gpu']='gpu'
) -> AnnData:
    """
    Perform super-resolution cell-type deconvolution using DINOv2-based method.

    This function:
        1. Uses a pre-trained DINOv2 model to enhance spatial resolution of
           cell-type deconvolution results.
        2. Integrates image features with spatial transcriptomics data.

    Parameters
    ----------
    adata_vis : AnnData
        Spatial transcriptomics AnnData object (e.g., Visium data) containing
        expression counts of spots or detected cells.
    deconv_adata : AnnData
        AnnData object containing initial deconvolution results.
    img_dir : str
        Directory path to the tissue image associated with `adata_vis`.
    neighb : int, optional
        Number of neighboring spots/cells to consider. Default is 3.
    radius : int, optional
        Radius for image patch extraction. Default is 129.
    epoch : int, optional
        Number of training epochs for the super-resolution model. Default is 50.
    batch_size : int, optional
        Batch size for training. Default is 32.
    num_workers : int, optional
        Number of worker threads for data loading. Default is 4.
    accelerator : {'cpu', 'gpu'}, optional
        Device to use for computation. Default is 'gpu'.

    Returns
    -------
    AnnData
        The input `adata_vis` with enhanced resolution deconvolution results.

    Raises
    ------
    ValueError
        If required parameters are missing or invalid.
    """
    
    logger.info("Running super-resolution cell-type deconvolution...")
    superres_fn = _import_backend("superres_core")
    
    sr_adata = superres_fn(
        deconv_adata=deconv_adata,
        adata_vis=adata_vis,
        img_dir=img_dir,
        neighb=neighb,
        radius=radius,
        epoch=epoch,
        batch_size=batch_size,
        num_workers=num_workers,
        accelerator=accelerator
    )

    return sr_adata

#     cta = CellTypeAnnotator(
#         spot_adata=deconv_adata,
#         sr_spot_adata=sr_deconv_adata,
#         seg_adata=segment_adata,
#         priori_type_affinities=None,  # 可选
#         alpha=0.3,
#         ot_mode="emd",      # "emd" or "sinkhorn"
#         # sinkhorn_reg=0.01,
#         # qp_solver="osqp",        # qpsolvers 开源后端
#         use_mip=True            # 如需精确 0/1 指派，设 True（需 Gurobi）
#     )

#     cta.filter_and_build_affiliations()
#     cta.compute_counts_and_integerize()
#     if cta.mode == 'mor':
#         cta.build_type_transfer(factor=2.0)
#     seg_adata_pred = cta.infer_cell_types()
def celltype_annotator(
        decov_adata: AnnData,
        sr_deconv_adata: AnnData,
        seg_adata: AnnData,
        priori_type_affinities=None,
        alpha=0.3,
        ot_mode="emd",          # "sinkhorn" or "emd"
        sinkhorn_reg=0.01,           # Sinkhorn 正则
        qp_solver: str = "osqp",     # qpsolvers 的后端选择："osqp"|"cvxopt"|...
        use_mip: bool = False        # True 则用 Gurobi MILP 做最终 0/1 指派（若可用）
    ) -> AnnData:
    """ Annotate cell types for segmented cells using spatial transcriptomics data.

    This function integrates the deconvolution results with the segmentation information
    to assign cell types to each segment.

    Parameters
    ----------
    decov_adata : AnnData
        The deconvolution results.
    sr_deconv_adata : AnnData
        The super-resolved deconvolution results.
    seg_adata : AnnData
        The segmentation results.
    priori_type_affinities : Optional[Dict[str, float]]
        Prior affinities for cell type assignment.
    alpha : float
        Regularization parameter for cell type assignment.
    ot_mode : str
        The optimal transport mode to use ("sinkhorn" or "emd").
    sinkhorn_reg : float
        Regularization parameter for Sinkhorn distance.
    qp_solver : str
        The quadratic programming solver to use.
    use_mip : bool
        Whether to use mixed-integer programming for final assignment.

    Returns
    -------
    AnnData
        The annotated segmentation results.
    """
    logger.info("Running cell type annotation...")
    annotator = _import_backend("annotator_core")

    seg_adata_pred = annotator(
        spot_adata=decov_adata,
        sr_spot_adata=sr_deconv_adata,
        seg_adata=seg_adata,
        priori_type_affinities=priori_type_affinities,  # 可选
        alpha=alpha,
        ot_mode=ot_mode,      # "emd" or "sinkhorn"
        sinkhorn_reg=sinkhorn_reg,
        qp_solver=qp_solver,        # qpsolvers 开源后端
        use_mip=use_mip            # 如需精确 0/1 指派，设 True（需 Gurobi）
    )

    return seg_adata_pred


__all__ = ["deconv_celltype", "superres_celltype"]