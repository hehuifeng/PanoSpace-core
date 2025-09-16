from __future__ import annotations

from anndata import AnnData
from typing import Dict
import logging

from ._annotator_backend.annotator_utils import CellTypeAnnotator
from panospace._core import register

logger = logging.getLogger(__name__)

def annotator_core(
    spot_adata: AnnData,
    sr_spot_adata: AnnData,
    seg_adata: AnnData,
    priori_type_affinities: Dict[str, float] | None = None,
    alpha: float = 0.3,
    ot_mode: str = "emd",          # "sinkhorn" or "emd"
    sinkhorn_reg: float = 0.01,           # Sinkhorn 正则
    qp_solver: str = "osqp",     # qpsolvers 的后端选择："osqp"|"cvxopt"|...
    use_mip: bool = False        # True 则用 Gurobi MILP 做最终 0/1 指派（若可用）
) -> AnnData:
    """Annotate segmented cells with cell types using spot-level and super-resolved deconvolution results.

    Parameters
    ----------
    spot_adata
        AnnData object containing spot-level deconvolution results.
    sr_spot_adata
        AnnData object containing super-resolved spot-level deconvolution results.
    seg_adata
        AnnData object containing segmented cell data.
    priori_type_affinities
        Optional dictionary specifying prior affinities for cell types.
    alpha
        Weighting factor for combining information from spot-level and super-resolved data.
    ot_mode
        Optimal transport mode to use - either "sinkhorn" or "emd".
    sinkhorn_reg
        Regularization parameter for Sinkhorn algorithm (if used).
    qp_solver
        Quadratic programming solver to use - options include "osqp", "cvxopt", etc.
    use_mip
        If True, uses Gurobi MILP for final 0/1 assignment (if available).

    Returns
    -------
    AnnData
        The annotated segmentation results.
    """

    seg_adata_pred = CellTypeAnnotator(
        spot_adata=spot_adata,
        sr_spot_adata=sr_spot_adata,
        seg_adata=seg_adata,
        priori_type_affinities=priori_type_affinities,
        alpha=alpha,
        ot_mode=ot_mode,          # "sinkhorn" or "emd"
        sinkhorn_reg=sinkhorn_reg,           # Sinkhorn 正则
        qp_solver=qp_solver,        # qpsolvers 开源后端
        use_mip=use_mip            # 如需精确 0/1 指派，设 True（需 Gurobi）
    )
    seg_adata_pred.filter_and_build_affiliations()
    seg_adata_pred.compute_counts_and_integerize()
    if seg_adata_pred.mode == 'mor':
        seg_adata_pred.build_type_transfer(factor=2.0)
    seg_adata_pred = seg_adata_pred.infer_cell_types()

    return seg_adata_pred

# Register backend -------------------------------------------------------------
register("annotation", "annotator", annotator_core)