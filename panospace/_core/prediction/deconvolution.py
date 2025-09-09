"""panospace._core.prediction.deconvolution
================================================
Spot-level deconvolution utilities (stage-1 of expression prediction).

The function provided here splits the *spot* gene-expression matrix into a
three-way tensor **X̃** (spot x cell-type x gene) as described in
Eq.(2) of the manuscript::

    X̃_{jkg} =  X_{jg} · P_{jk} · μ̂_{kg} / ∑_{k'} P_{jk'} μ̂_{k'g}

where

* *j* indexes spots, *k* cell types, *g* genes;
* *X* is the observed spot-level count matrix;
* *P* = *Y* is the cell-type proportion matrix obtained from EnDecon or other
  deconvolution tool;
* μ̂ is the mean expression profile per cell type estimated from the reference
  scRNA-seq atlas.

The resulting tensor can subsequently be consumed by graph-based SSL or other
super-resolution algorithms.

Notes
-----
* The function is **CPU-only** and pure NumPy/Pandas to keep dependencies
  minimal; GPU-accelerated versions can wrap this logic inside CuPy or
  PyTorch.  All heavy lifting is vectorised and reasonably fast for typical
  Visium sizes (<10 K spots x 20 K genes).
* We do **not** register this module with ``panospace._core.register`` because
  it is an internal utility rather than a user-selectable back-end.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

__all__: Tuple[str, ...] = ("deconvolve_spots",)

logger = logging.getLogger("panospace._core.prediction.deconvolution")


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def _validate_inputs(
    sdata: "SpatialData", ref_adata: "AnnData", props: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Validate shapes & genes; return aligned arrays ready for numpy ops."""

    # Extract spot counts and barcodes
    try:
        spot_table = sdata.tables["spots"]  # type: ignore[index]
    except KeyError as e:
        raise KeyError("`sdata.tables['spots']` missing – load Visium spots first") from e

    spot_counts = spot_table.X  # (n_spots × n_genes) SciPy sparse or np.ndarray
    if not isinstance(spot_counts, np.ndarray):
        spot_counts = spot_counts.toarray()
    barcodes = spot_table.obs_names.to_numpy()

    # Align proportion matrix rows to barcodes
    if not np.array_equal(barcodes, props.index.to_numpy()):
        props = props.loc[barcodes]

    # Check gene sets
    genes_sdata = spot_table.var_names
    genes_ref = ref_adata.var_names
    if not genes_sdata.equals(genes_ref):
        # Intersect & reorder to shared genes
        shared = genes_sdata.intersection(genes_ref)
        logger.warning(
            "Gene sets differ between SpatialData spots and reference – using %d shared genes",
            shared.size,
        )
        spot_counts = spot_counts[:, genes_sdata.get_indexer(shared)]
        ref_adata = ref_adata[:, shared]
        genes_sdata = shared

    # Compute μ̂ (mean expr per cell type)
    ct_means = (
        ref_adata.to_df()
        .groupby(ref_adata.obs["cell_type"], observed=True)
        .mean()
        .to_numpy()
    )
    cell_types = ref_adata.obs["cell_type"].unique()
    if not np.array_equal(cell_types, props.columns.to_numpy()):
        # Re-order/align columns
        ct_means = ct_means[
            np.searchsorted(cell_types, props.columns.to_numpy())  # type: ignore[arg-type]
        ]
        cell_types = props.columns.to_numpy()

    return props, spot_counts, ct_means, genes_sdata.to_numpy()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def deconvolve_spots(
    sdata: "SpatialData",
    ref_adata: "AnnData",
    proportion_df: pd.DataFrame,
    *,
    target_library_size: float | None = 10_000.0,
    log1p: bool = False,
    dtype: str | np.dtype = "float32",
) -> np.ndarray:
    """Deconvolve spot counts into spotxcell-typexgene tensor.

    Parameters
    ----------
    sdata
        A :class:`spatialdata.SpatialData` object with a ``spots`` table whose
        ``.X`` stores raw or normalised counts.
    ref_adata
        Reference scRNA-seq with *obs['cell_type']* column.
    proportion_df
        Deconvolution output of shape (n_spots, n_cell_types), rows indexed by
        Visium barcodes.
    target_library_size
        If not *None*, each pseudo-cell (Γ splitting) is normalised to this
        library size *before* optional ``log1p`` transform.
    log1p
        Apply ``np.log1p`` to the deconvolved counts.
    dtype
        Final NumPy dtype of the returned array.

    Returns
    -------
    np.ndarray
        Array with shape (n_spots, n_cell_types, n_genes) in the same gene
        order as ``ref_adata.var_names``.
    """

    t0 = time.perf_counter()
    props, spot_counts, ct_means, genes = _validate_inputs(
        sdata, ref_adata, proportion_df
    )

    P = props.to_numpy(dtype=np.float32)  # (n_spots, n_ct)
    X = spot_counts.astype(np.float32)
    MU = ct_means.astype(np.float32)  # (n_ct, n_genes)

    logger.info(
        "Deconvolving %s spots × %s genes into %s cell types…",
        X.shape[0], X.shape[1], P.shape[1],
    )

    # Pre-compute denominator per spot/gene
    denom = (P @ MU)  # (n_spots, n_genes)
    # Avoid division by zero – add tiny epsilon
    denom[denom == 0.0] = 1e-6

    # Compute tensor – vectorised broadcasting
    tensor = (X[:, None, :] * P[:, :, None] * MU[None, :, :]) / denom[:, None, :]

    # Normalisation to target library size
    if target_library_size is not None:
        lib_scale = target_library_size / tensor.sum(axis=2, keepdims=True)
        tensor *= lib_scale

    if log1p:
        np.log1p(tensor, out=tensor)

    tensor = tensor.astype(dtype, copy=False)

    logger.info("Finished deconvolution (%.2f s)", time.perf_counter() - t0)

    return tensor
