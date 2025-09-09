"""panospace.tl.microenv
=======================
High-level wrapper for **spatial micro-environment analysis**.  It quantifies
how the neighbourhood of a *sender* cell type influences gene expression in a
*receiver* cell type and can optionally infer ligand-receptor-target (L-R-T)
networks.

All heavy computation is delegated to specialised back-end implementations in
:pydata:`panospace._core.microenv`.  This wrapper only performs:

1. **Input validation** - detected cells + cell-type labels + predicted
   expression must exist.
2. **Parameter normalisation** - convert paths/AnnData into
   :class:`spatialdata.SpatialData`, check that the requested cell types exist
   and that a prediction layer is available.
3. **Backend dispatch** - call the selected algorithm and collect results.
4. **Result storage** - write outputs into ``sdata.uns['panospace/microenv']``
   for downstream plotting and return a typed :class:`MicroEnvResult` when
   requested.

Example
-------
>>> import panospace as ps
>>> sdata = ps.io.read_visium("tumour/")
>>> sdata = ps.tl.detect_cells(sdata)
>>> sdata = ps.tl.annotate_celltype(sdata, scrna_ref="atlas.h5ad")
>>> sdata = ps.tl.predict_expr(sdata)
>>> sdata, res = ps.tl.microenv_analysis(
...     sdata,
...     sender="CAF",
...     receiver="Cancer_Epi",
...     radius=50,  # µm
...     top_n_genes=200,
...     infer_lr=True,
...     return_results=True,
... )
>>> res.gene_table.head()

"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, TYPE_CHECKING

import pandas as pd

from ..io.adapters import _ensure_spatialdata
from ..io._schemas import SCHEMA_REGISTRY

if TYPE_CHECKING:  # pragma: no cover
    import spatialdata as spd

logger = logging.getLogger("panospace.tl")

# -----------------------------------------------------------------------------
# Back-end registry
# -----------------------------------------------------------------------------

_BACKENDS: dict[str, str] = {
    "default": "panospace._core.microenv.stat_corr:microenv_core",
    "lrnet": "panospace._core.microenv.lrnet:lr_network_core",
}


def _import_backend(entry: str):
    """Lazy import a *module:function* string."""
    mod_name, func_name = entry.split(":")
    module = __import__(mod_name, fromlist=[func_name])
    return getattr(module, func_name)


# -----------------------------------------------------------------------------
# Public dataclass & API
# -----------------------------------------------------------------------------

@dataclass
class MicroEnvResult:
    """Container returned when *return_results=True*."""

    gene_table: pd.DataFrame
    lr_table: Optional[pd.DataFrame] = None


# ──────────────────────────────────────────────────────────────────────────────

def microenv_analysis(
    sdata: "spd.SpatialData | str | Path",  # _ensure_spatialdata handles conversion
    *,
    sender: str,
    receiver: str,
    radius: float = 50.0,
    top_n_genes: int | None = 100,
    infer_lr: bool = False,
    method: Literal["default"] = "default",
    overwrite: bool = False,
    backend_kwargs: Optional[dict] = None,
    return_results: bool = False,
) -> "spd.SpatialData | Tuple[spd.SpatialData, MicroEnvResult]":
    """Run neighbourhood-gene-expression correlation and optional L-R-T network.

    Parameters
    ----------
    sdata
        A :class:`spatialdata.SpatialData` object or something that can be
        converted into one (path, AnnData, …).
    sender, receiver
        Cell-type labels present in ``sdata.tables['cells'].obs['cell_type']``.
    radius
        Search radius *in microns* for counting sender neighbours.
    top_n_genes
        Number of receiver genes (highest |correlation|) to keep. ``None``
        keeps all.
    infer_lr
        If ``True`` a second backend infers ligand-receptor-target networks.
    method
        Key of the backend registry - currently only ``"default"`` is mapped.
    overwrite
        Overwrite existing results in ``uns['panospace/microenv']``.
    backend_kwargs
        Extra kwargs forwarded to backend function(s).
    return_results
        Also return a :class:`MicroEnvResult` alongside the modified object.

    Returns
    -------
    sdata
        The updated :class:`spatialdata.SpatialData`.
    (optional) MicroEnvResult
        Only when ``return_results=True``.
    """

    t0 = time.time()
    backend_kwargs = backend_kwargs or {}

    # ------------------------------------------------------------------
    # Normalise & validate input
    # ------------------------------------------------------------------
    sdata = _ensure_spatialdata(sdata)

    cells = sdata.tables.get("cells")
    if cells is None:
        raise ValueError("Cells table missing - run ps.tl.detect_cells() first.")

    if "cell_type" not in cells.obs.columns:
        raise ValueError("Cell types missing - run ps.tl.annotate_celltype() first.")

    layer_name = "expr_pred"
    if layer_name not in sdata.layers:
        raise ValueError("Predicted expression missing - run ps.tl.predict_expr() first.")

    if sender not in cells.obs["cell_type"].unique() or receiver not in cells.obs["cell_type"].unique():
        raise ValueError(
            f"Sender or receiver type not found in cell_type column: {sender}, {receiver}."
        )

    if radius <= 0:
        raise ValueError("radius must be > 0 µm.")

    SCHEMA_REGISTRY["cells"].validate(cells)

    if not overwrite and "panospace/microenv" in sdata.uns:
        raise ValueError("Analysis already exists. Set overwrite=True to recompute.")

    # ------------------------------------------------------------------
    # Backend dispatch
    # ------------------------------------------------------------------
    core_fn = _import_backend(_BACKENDS[method])
    logger.info(
        "[microenv] sender=%s, receiver=%s, radius=%.1f µm, top_n=%s, infer_lr=%s",
        sender,
        receiver,
        radius,
        str(top_n_genes),
        infer_lr,
    )

    gene_tbl = core_fn(
        sdata,
        sender=sender,
        receiver=receiver,
        radius=radius,
        top_n=top_n_genes,
        layer=layer_name,
        **backend_kwargs,
    )

    lr_tbl = None
    if infer_lr:
        lr_fn = _import_backend(_BACKENDS["lrnet"])
        lr_tbl = lr_fn(
            sdata,
            sender=sender,
            receiver=receiver,
            gene_table=gene_tbl,
            **backend_kwargs,
        )

    # ------------------------------------------------------------------
    # Store & finish
    # ------------------------------------------------------------------
    sdata.uns["panospace/microenv"] = {
        "sender": sender,
        "receiver": receiver,
        "radius": radius,
        "top_n_genes": top_n_genes,
        "gene_table": gene_tbl,
        "lr_table": lr_tbl,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    logger.info(
        "[microenv] finished in %.2f s (genes=%d, lr=%s)",
        time.time() - t0,
        len(gene_tbl),
        lr_tbl is not None,
    )

    if return_results:
        return sdata, MicroEnvResult(gene_table=gene_tbl, lr_table=lr_tbl)
    return sdata


__all__: Sequence[str] = ["microenv_analysis", "MicroEnvResult"]
