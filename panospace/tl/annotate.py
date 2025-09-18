"""panospace.tl.annotate
========================
High-level interface for **cell-type annotation** and deconvolution at
single-cell resolution in spatial transcriptomics.

This module provides three main functionalities:
    1. Perform cell-type deconvolution of spatial data using multiple
       backends (e.g., RCTD, cell2location, spatialDWLS).
    2. Integrate backend results into an ensemble consensus using
       the EnDecon algorithm.
    3. Refine and project cell-type assignments to super-resolved
       or segmented cell units.

Backends
--------
Supported gene-expression-based deconvolution backends:
* ``"RCTD"``          - Robust Cell Type Decomposition.
* ``"cell2location"`` - Probabilistic cell-type mapping model.
* ``"spatialDWLS"``   - Deconvolution via dampened weighted least squares.
* ``"endecon"``       - Ensemble integration of multiple backends.


"""

from __future__ import annotations

import logging
from typing import List, Literal

import pandas as pd
from anndata import AnnData

logger = logging.getLogger("panospace.tl")

from . import _import_backend


# -----------------------------------------------------------------------------
# Cell-type deconvolution
# -----------------------------------------------------------------------------
def deconv_celltype(
    adata_vis: AnnData,
    sc_adata: AnnData,
    celltype_key: str,
    methods: List[Literal["RCTD", "cell2location", "spatialDWLS"]] = ["RCTD", "cell2location", "spatialDWLS"],
) -> AnnData:
    """
    Run cell-type deconvolution on spatial transcriptomics data.

    Steps
    -----
    1. Execute each selected backend on the input data.
    2. Collect per-backend results and align them by common cells.
    3. Integrate results into an ensemble consensus with EnDecon.

    Parameters
    ----------
    adata_vis : AnnData
        Spatial transcriptomics AnnData object (e.g., Visium data).
    sc_adata : AnnData
        Reference single-cell RNA-seq dataset with known cell-type labels.
    celltype_key : str
        Column in ``sc_adata.obs`` specifying cell-type annotations.
    methods : list of {"RCTD", "cell2location", "spatialDWLS"}, optional
        Backends to run. Default includes all three.

    Returns
    -------
    AnnData
        A copy of ``adata_vis`` restricted to common cells, with results
        stored in:
        - ``uns["X_deconv_<backend>"]`` : per-backend results (DataFrame).
        - ``uns["X_deconv_ensemble"]`` : EnDecon consensus with probability
          matrix ``H_norm``.

    Raises
    ------
    KeyError
        If ``celltype_key`` is missing in ``sc_adata.obs``.
    """
    if celltype_key not in sc_adata.obs:
        raise KeyError(f"Reference AnnData must contain '{celltype_key}' in .obs")

    celltype = sc_adata.obs[celltype_key].unique().tolist()
    celltype.sort()
    adata_vis.uns["celltype"] = celltype
    results_list = []
    # import pdb
    for method in methods:
        logger.info("Running cell-type deconvolution with backend '%s'", method)
        backend_fn = _import_backend(method)

        result_df: pd.DataFrame = backend_fn(
            sc_adata=sc_adata,
            adata_vis=adata_vis,
            celltype_key=celltype_key,
        )
        adata_vis.uns[f"X_deconv_{method}"] = result_df
        results_list.append(result_df)
        # pdb.set_trace()

    # Align results by shared cells
    common_index = sorted(set.intersection(*(set(df.index) for df in results_list)))
    aligned_results = [
        df.loc[common_index, celltype].to_numpy(dtype=float) for df in results_list
    ]

    logger.info("Running EnDecon ensemble integration...")
    EnDecon = _import_backend("endecon")
    ensemble_result = EnDecon(aligned_results)

    ensemble_result["H_norm"] = pd.DataFrame(
        ensemble_result["H_norm"], index=common_index, columns=celltype
    )

    deconv_adata = adata_vis.copy()[common_index].copy()
    deconv_adata.obs = deconv_adata.obs.join(ensemble_result["H_norm"], how="left")

    return deconv_adata


# -----------------------------------------------------------------------------
# Super-resolution refinement
# -----------------------------------------------------------------------------
def superres_celltype(
    deconv_adata: AnnData,
    img_dir: str,
    neighb: int = 3,
    radius: int = 129,
    epoch: int = 50,
    class_weights=None,
    learning_rate: float = 1e-4,
    local_path: str = "~/.panospace_cache/dinov2-base",
    pretrained_model_name: str = "facebook/dinov2-base",
    cache_dir: str = "~/.panospace_cache",
    batch_size: int = 32,
    num_workers: int = 4,
    accelerator: Literal["cpu", "gpu"] = "gpu",
) -> AnnData:
    """
    Refine deconvolution results at higher spatial resolution using DINOv2.

    Parameters
    ----------
    deconv_adata : AnnData
        AnnData object with initial deconvolution results.
    img_dir : str
        Path to tissue image corresponding to ``deconv_adata``.
    neighb : int, optional
        Number of neighboring spots/cells considered (default: 3).
    radius : int, optional
        Radius for image patch extraction (default: 129).
    epoch : int, optional
        Number of training epochs (default: 50).
    batch_size : int, optional
        Batch size for training (default: 32).
    num_workers : int, optional
        Number of data loader workers (default: 4).
    accelerator : {"cpu", "gpu"}, optional
        Compute device (default: "gpu").

    Returns
    -------
    AnnData
        ``adata_vis`` with super-resolved deconvolution outputs.
    """
    logger.info("Running super-resolution deconvolution...")
    superres_fn = _import_backend("superres_core")

    sr_adata = superres_fn(
        deconv_adata=deconv_adata,
        img_dir=img_dir,
        neighb=neighb,
        radius=radius,
        class_weights=class_weights,
        learning_rate=learning_rate,
        local_path=local_path,
        pretrained_model_name=pretrained_model_name,
        cache_dir=cache_dir,
        epoch=epoch,
        batch_size=batch_size,
        num_workers=num_workers,
        accelerator=accelerator,
    )
    return sr_adata


# -----------------------------------------------------------------------------
# Cell-type annotation of segments
# -----------------------------------------------------------------------------
def celltype_annotator(
    decov_adata: AnnData,
    sr_deconv_adata: AnnData,
    seg_adata: AnnData,
    priori_type_affinities=None,
    alpha: float = 0.3,
    ot_mode: str = "emd",
    sinkhorn_reg: float = 0.01,
    qp_solver: str = "osqp",
    use_mip: bool = False,
) -> AnnData:
    """
    Assign cell types to segmented cells using deconvolution outputs.

    Combines spot-level deconvolution, super-resolved refinement,
    and segmentation boundaries to obtain consistent cell-type
    assignments via optimal transport.

    Parameters
    ----------
    decov_adata : AnnData
        Spot-level deconvolution results.
    sr_deconv_adata : AnnData
        Super-resolved deconvolution results.
    seg_adata : AnnData
        Segmentation results (cell masks/coordinates).
    priori_type_affinities : dict, optional
        Prior affinities between cell types.
    alpha : float, optional
        Regularization strength (default: 0.3).
    ot_mode : {"sinkhorn", "emd"}, optional
        Optimal transport variant (default: "emd").
    sinkhorn_reg : float, optional
        Sinkhorn regularization coefficient (default: 0.01).
    qp_solver : {"osqp", "cvxopt", ...}, optional
        Quadratic programming solver (default: "osqp").
    use_mip : bool, optional
        If True, use Gurobi MILP for exact 0/1 assignments (default: False).

    Returns
    -------
    AnnData
        Segmentation AnnData with cell-type annotations.
    """
    logger.info("Running cell-type annotation for segmented cells...")
    annotator = _import_backend("annotator_core")

    seg_adata_pred = annotator(
        spot_adata=decov_adata,
        sr_spot_adata=sr_deconv_adata,
        seg_adata=seg_adata,
        priori_type_affinities=priori_type_affinities,
        alpha=alpha,
        ot_mode=ot_mode,
        sinkhorn_reg=sinkhorn_reg,
        qp_solver=qp_solver,
        use_mip=use_mip,
    )
    return seg_adata_pred


__all__ = ["deconv_celltype", "superres_celltype", "celltype_annotator"]
