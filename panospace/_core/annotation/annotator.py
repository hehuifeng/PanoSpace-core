from __future__ import annotations

from anndata import AnnData
from typing import Dict
import logging

from ._annotator_backend.annotator_utils import CellTypeAnnotator

logger = logging.getLogger(__name__)

def annotator_core(
    spot_adata: AnnData,
    sr_spot_adata: AnnData,
    seg_adata: AnnData,
    priori_type_affinities: Dict[str, float] | None = None,
    alpha: float = 0.3,
    ot_mode: str = "emd",          # "sinkhorn" or "emd"
    sinkhorn_reg: float = 0.01     # Sinkhorn regularization
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
        Fusion weight between spatial propagation and morphology prior.
    ot_mode
        Optimal transport mode to use - either "sinkhorn" or "emd".
    sinkhorn_reg
        Regularization parameter for Sinkhorn algorithm (if used).

    Returns
    -------
    AnnData
        The annotated segmentation results.

    Notes
    -----
    The final assignment uses Mixed Integer Linear Programming (MILP) with:
      - Gurobi (if available, fastest)
      - SCIP (open-source fallback)
    Both solvers produce mathematically equivalent results.
    """

    _seg_adata_pred = CellTypeAnnotator(
        spot_adata=spot_adata,
        sr_spot_adata=sr_spot_adata,
        seg_adata=seg_adata,
        priori_type_affinities=priori_type_affinities,
        alpha=alpha,
        ot_mode=ot_mode,                # "sinkhorn" or "emd"
        sinkhorn_reg=sinkhorn_reg       # Sinkhorn regularization
    )
    _seg_adata_pred.filter_and_build_affiliations()
    _seg_adata_pred.compute_counts_and_integerize()
    if _seg_adata_pred.mode == 'mor':
        _seg_adata_pred.build_type_transfer(factor=2.0)
    seg_adata_pred = _seg_adata_pred.infer_cell_types()

    return seg_adata_pred, _seg_adata_pred
