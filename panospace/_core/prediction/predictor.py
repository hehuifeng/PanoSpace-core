"""panospace._core.prediction.predictor
=======================================

Core entry point that wires the sparse predictor backend into the public API.
It prepares inputs, triggers per cell-type spot decomposition, runs diffusion,
and returns an AnnData with inferred expression for all target nuclei.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from anndata import AnnData

from .predictor_backend.predictor_utils import GeneExpPredictor

logger = logging.getLogger(__name__)


def predictor_core(
    sc_adata: AnnData,
    spot_adata: AnnData,
    infered_adata: AnnData,
    celltype_list: List[str],
    celltype_column: str = "celltype_major",
) -> AnnData:
    """Run the sparse prediction pipeline end-to-end.

    Steps
    -----
    1) Instantiate the sparse backend with the three input objects
       (single-cell reference, spatial spots, and nuclei to infer).
    2) Decompose spot expression into per–cell-type contributions using the
       provided list and the label column in the single-cell reference.
    3) Diffuse those signals to nuclei coordinates with a random-walk graph and
       early stopping.

    Parameters
    ----------
    sc_adata
        Single-cell reference; used to compute per–cell-type mean profiles.
    spot_adata
        Spatial transcriptomics data with expression in ``.X``, coordinates in
        ``.obsm['spatial']``, and ``.uns['radius']`` for neighbourhood queries.
    infered_adata
        Nuclei whose expression will be inferred. Must include coordinates in
        ``.obsm['spatial']`` and a predicted type in ``.obs['pred_cell_type']``.
    celltype_list
        Ordered list of cell types to process. The same order is assumed for
        mixture columns in ``spot_adata.obs``.
    celltype_column
        Column name in ``sc_adata.obs`` containing the cell-type labels used
        for computing type means.

    Returns
    -------
    AnnData
        An object where rows correspond to inferred nuclei and columns to genes,
        with the inferred expression stored in ``.X``. The ``.obs`` table
        preserves ``pred_cell_type`` and coordinates are available in
        ``.obsm['spatial']``.

    Notes
    -----
    Default diffusion hyperparameters are chosen to be conservative and match
    typical settings for stable convergence on medium-sized datasets.
    """
    start_time = time.time()

    # Initialize the backend
    predictor = GeneExpPredictor(sc_adata, spot_adata, infered_adata)

    # Build per–cell-type spot matrices (normalised and log1p-transformed)
    predictor.compute_celltype_specific_spot_expression(celltype_list, celltype_column)

    # Run diffusion to obtain nucleus-level expression
    adata_pred = predictor.infer_expression(
        gamma=0.1,      # soft constraint toward labeled initialisation
        iterations=20,  # maximum number of diffusion steps
        tol=1e-4,       # early-stopping tolerance on mean squared change
        patience=5      # allowed non-improving iterations
    )

    logger.info("Predictor core completed in %.2f seconds.", time.time() - start_time)
    return adata_pred
